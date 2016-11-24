#include "GPUFilter.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "settings.h"
#include "check_error.h"


namespace GPUFilter {

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


    __device__
    void prescan (int tid, int *s_cache, int *g_prescan_out, int *g_block_sums_out)
    {
        // -- PRESCAN -- //
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
            // First save the total sum to the output block sums array
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


        // Copy data to the output array
        g_prescan_out[2*tid]   = s_cache[2*threadIdx.x];
        g_prescan_out[2*tid+1] = s_cache[2*threadIdx.x+1];
    }


    /**
     * @brief Filters the data array and computes the indices (in each block) of the filtered elements using prescan
     * @param g_data_array_in Array of data to be filtered
     * @param length Length of the array of data
     * @param g_prescan_out Indices of the filtered elements (per block)
     * @param g_block_sums_out Numbers of filtered elements in each block
     */
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
        prescan(tid, s_cache, g_prescan_out, g_block_sums_out);
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
        prescan(tid, s_cache, g_prescan_out, g_block_sums_out);
    }


    __global__
    void propagateSum (int *g_level_top, int *g_level_bottom, int top_level_size)
    {
//        int bottom_level_size = top_level_size*2*THREADS_PER_BLOCK;
        int tid = blockIdx.x*blockDim.x + threadIdx.x;

        int top_id = tid / (2*THREADS_PER_BLOCK);

        g_level_bottom[tid] += g_level_top[top_id];
    }


    void determineIndicesRecursive (std::vector<int*> &g_index_pyramid, int level, int level_size)
    {
        int num_blocks = std::ceil(level_size / (2.0*THREADS_PER_BLOCK));

        int* g_block_sums_out;
        cudaMalloc((void**)&g_block_sums_out, num_blocks*sizeof(int));

        onlyPrescan<<<num_blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK>>>(g_index_pyramid[level],
                                                                            level_size,
                                                                            g_index_pyramid[level],
                                                                            g_block_sums_out);

        g_index_pyramid.push_back(g_block_sums_out);

        if (num_blocks > 1)
        {
            // Call the recursive function
            determineIndicesRecursive(g_index_pyramid, level+1, num_blocks);
        }
    }


    void determineIndices (const DataArray &da, int *g_indices_out, int &output_size_out)
    {
        std::vector<int*> g_index_pyramid;

        // We use only one dimensional indices of grid cells and blocks because it is easier - we have a linear
        // vector of data
        // Each block can process double the amount of data than the number of threads in it
        int num_blocks = std::ceil(da.size / (2.0*THREADS_PER_BLOCK));

        // Copy data to device
        Data* g_data_array_in;
        cudaMalloc((void**)&g_data_array_in, da.size*sizeof(Data));
        cudaMemcpy(g_data_array_in, da.array, da.size*sizeof(Data), cudaMemcpyHostToDevice);

        int* g_prescan_out;
        cudaMalloc((void**)&g_prescan_out, da.size*sizeof(int));
        int* g_block_sums_out;
        cudaMalloc((void**)&g_block_sums_out, num_blocks*sizeof(int));

        filterPrescan<<<num_blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK>>>(g_data_array_in, da.size,
                                                                              g_prescan_out, g_block_sums_out);

        g_index_pyramid.push_back(g_prescan_out);
        g_index_pyramid.push_back(g_block_sums_out);

        if (num_blocks > 1)
        {
            // Call the recursive function
            determineIndicesRecursive(g_index_pyramid, 1, num_blocks);
        }


        cudaMemcpy(&output_size_out, g_index_pyramid[g_index_pyramid.size()-1], sizeof(int), cudaMemcpyDeviceToHost);


        int level_size = 1;
        for (int l = g_index_pyramid.size()-2; l > 0; --l)
        {
             int block_sums_out[level_size];
                cudaMemcpy(block_sums_out, g_index_pyramid[l], level_size*sizeof(int), cudaMemcpyDeviceToHost);

                for (int i = 0; i < level_size; ++i) std::cout << block_sums_out[i] << std::endl;
                std::cout << std::endl;





//            propagateSum<<<2*level_size, THREADS_PER_BLOCK>>>(g_index_pyramid[l], g_index_pyramid[l-1],
//                                                              level_size);

            level_size = level_size * 2*THREADS_PER_BLOCK;

//            if (l != 0) cudaFree(g_index_pyramid[l]);
        }


        g_indices_out = g_index_pyramid[0];
    }
}


DataArray filterArray (const DataArray &da)
{
    std::cout << "Filtering data with CUDA!!" << std::endl;

    // We use only one dimensional indices of grid cells and blocks because it is easier - we have a linear
    // vector of data
    // Each block can process double the amount of data than the number of threads in it
    int num_blocks = std::ceil(da.size / (2.0*THREADS_PER_BLOCK));

    int* g_indices_out;
    int output_size;
    determineIndices(da, g_indices_out, output_size);

    int out[da.size];
    cudaMemcpy(out, g_indices_out, da.size*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < da.size; ++i)
    {
        std::cout << out[i] << std::endl;
    }

    std::cout << "Output size: " << output_size << std::endl;


//    // Copy data to gpu
//    Data* g_data_array_in;
//    cudaMalloc((void**)&g_data_array_in, da.size*sizeof(Data));
//    cudaMemcpy(g_data_array_in, da.array, da.size*sizeof(Data), cudaMemcpyHostToDevice);

//    // Output data vector
//    int* g_prescan_out;
//    cudaMalloc((void**)&g_prescan_out, da.size*sizeof(int));
//    int* g_block_sums_out;
//    cudaMalloc((void**)&g_block_sums_out, num_blocks*sizeof(int));


//    filterPrescan<<<num_blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK>>>(g_data_array_in, da.size,
//                                                                    g_prescan_out, g_block_sums_out);


//    int prescan_out[da.size];
//    cudaMemcpy(prescan_out, g_prescan_out, da.size*sizeof(int), cudaMemcpyDeviceToHost);
//    int block_sums_out[num_blocks];
//    cudaMemcpy(block_sums_out, g_block_sums_out, num_blocks*sizeof(int), cudaMemcpyDeviceToHost);

//    cudaFree(g_data_array_in);
//    cudaFree(g_prescan_out);


//    for (int i = 0; i < da.size; ++i)
//    {
//        std::cout << da.array[i].key << ": " << ((da.array[i].key >= FILTER_MIN && da.array[i].key <= FILTER_MAX) ? 1 : 0) << " " << prescan_out[i] << std::endl;
//    }

//    for (int i = 0; i < num_blocks; ++i)
//    {
//        std::cout << block_sums_out[i] << std::endl;
//    }


    return DataArray(10);
}


bool initialize ()
{
    // Find out if there is a CUDA capable device
    int device_count;
    CHECK_ERROR(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        // Error, we cannot initialize
        return false;
    }
    else
    {
        // Copying a dummy to the device will initialize it
        int* gpu_dummy;
        cudaMalloc((void**)&gpu_dummy, sizeof(int));
        cudaFree(gpu_dummy);

        // Get properties of the device
        cudaDeviceProp device_properties;
        CHECK_ERROR(cudaGetDeviceProperties(&device_properties, 0));

        std::cout << "--------------------------------------------------------------" << std::endl;
        std::cout << "Device name:           " << device_properties.name << std::endl;
        std::cout << "Compute capability:    " << device_properties.major << "." << device_properties.minor << std::endl;
        std::cout << "Total global memory:   " << device_properties.totalGlobalMem << std::endl;
        std::cout << "Multiprocessor count:  " << device_properties.multiProcessorCount << std::endl;
        std::cout << "Max threads per block: " << device_properties.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dim:       " << device_properties.maxThreadsDim[0] << std::endl;
        std::cout << "Max grid size:         " << device_properties.maxGridSize[0] << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;

        return true;
    }
}


}


