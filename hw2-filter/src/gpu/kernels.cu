#include "kernels.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "settings.h"
#include "data.h"


namespace {

    /**
     * @brief Returns true if the key is in the interval, false if it should be thrown away
     * @param key
     * @return
     */
    __device__
    bool filter (int key)
    {
        return key >= FILTER_MIN && key <= FILTER_MAX;
    }


    /**
     * @brief Prescan algorithm (exclusive scan) implementation for GPU
     * @param s_cache Array that we want to prescan (size is 2*blockDim.x, blockDim.x has to be a power of 2!)
     * @param g_block_sums_out Array of block sums
     */
    __device__
    void prescan (int *s_cache, int *g_block_sums_out)
    {
        // The implementation is divided into upsweep and downsweep phase (see the page
        // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html for details)

        // Upsweep phase
        int spread = 1;
        for (int i = blockDim.x; i > 0; i >>= 1)
        {
            __syncthreads();

            if (threadIdx.x < i)
            {
                int idx_first  = ((2*threadIdx.x+1) * spread) - 1;
                int idx_second = ((2*threadIdx.x+2) * spread) - 1;
                s_cache[idx_second] += s_cache[idx_first];
            }

            spread <<= 1; // Multiply by 2
        }

        // We do not need to call __syncthreads() here because the last working thread was thread 0 and we
        // are now using the thread 0 to clear the last element -> no incorrect synchronization can happen

        // Set last element to 0 before the downsweep phase
        if (threadIdx.x == 0)
        {
            // First save the total sum to the output block sums array (we will further run prescan on it
            // as well)
            g_block_sums_out[blockIdx.x] = s_cache[2*blockDim.x-1];
            // Now clear the value
            s_cache[2*blockDim.x-1] = 0;
        }

        // Downsweep phase
        for (int i = 1; i <= blockDim.x; i <<= 1)
        {
            // This must be before the computation because the last operation in upsweep multiplied it even
            // more (out of bounds)
            spread >>= 1; // Divide by 2

            __syncthreads();

            if (threadIdx.x < i)
            {
                int idx_first  = ((2*threadIdx.x+1) * spread) - 1;
                int idx_second = ((2*threadIdx.x+2) * spread) - 1;

                int tmp = s_cache[idx_second];

                s_cache[idx_second] += s_cache[idx_first];  // Set the right child to L+current
                s_cache[idx_first]   = tmp;               // Set the left child to current
            }
        }

        __syncthreads();
    }

}


// //////////////////////////////////////////////////////////////////////////////////////////////////////// //
// --------------------------------------------  CUDA KERNELS  -------------------------------------------- //
// //////////////////////////////////////////////////////////////////////////////////////////////////////// //

__global__
void filterPrescan (Data *g_data_array_in, int length, int *g_prescan_out, int *g_block_sums_out)
{
    // The ID of the thread - we use only one dimensional grid and blocks
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    extern __shared__ int s_cache[];

    // For each Data cell determine whether it will be in the output array or not
    // We can process double the number of cells than threads - each thread reads 2 cells
    s_cache[2*threadIdx.x]   = filter(g_data_array_in[2*tid].key);
    s_cache[2*threadIdx.x+1] = filter(g_data_array_in[2*tid+1].key);

    // Perform the prescan
    prescan(s_cache, g_block_sums_out);

    // Copy data to the output array
    g_prescan_out[2*tid]   = s_cache[2*threadIdx.x];
    g_prescan_out[2*tid+1] = s_cache[2*threadIdx.x+1];
}


__global__
void onlyPrescan (int *g_array_in, int length, int *g_prescan_out, int *g_block_sums_out)
{
    // The ID of the thread - we use only one dimensional grid and blocks
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    extern __shared__ int s_cache[];

    // Copy each cell into shared memory
    s_cache[2*threadIdx.x]   = g_array_in[2*tid];
    s_cache[2*threadIdx.x+1] = g_array_in[2*tid+1];

    // Perform the prescan
    prescan(s_cache, g_block_sums_out);

    // Copy data to the output array
    g_prescan_out[2*tid]   = s_cache[2*threadIdx.x];
    g_prescan_out[2*tid+1] = s_cache[2*threadIdx.x+1];
}


__global__
void propagateSum (int *g_level_top, int *g_level_bottom, int top_level_size)
{
//        int bottom_level_size = top_level_size*2*THREADS_PER_BLOCK;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    int top_id = tid / (2*THREADS_PER_BLOCK);

    g_level_bottom[tid] += g_level_top[top_id];
}


__global__
void copyElementsToOutput (Data *g_da, int *g_indices, Data *g_da_out)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (filter(g_da[tid].key)) g_da_out[g_indices[tid]] = g_da[tid];
}


