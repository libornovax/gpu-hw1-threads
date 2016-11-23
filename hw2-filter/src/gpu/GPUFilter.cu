#include "GPUFilter.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
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


//    __global__
//    void filterArray (Data *data_array_in, int length, Data *data_array_out)
//    {
//        int threadId = threadIdx.x;
//    }


    __global__
    void prescan (Data *data_array_in, int length, int *data_prescan_out)
    {
        // The ID of the thread - we use only one dimensional grid and blocks
        int tid = blockIdx.x*blockDim.x + threadIdx.x;

        // For each Data cell determine whether it will be in the output array or not
        // We can process double the number of cells than threads - each thread reads 2 cells
        __shared__ int cache[2*THREADS_PER_BLOCK];

        cache[2*threadIdx.x]   = filter(data_array_in[2*tid].key);
        cache[2*threadIdx.x+1] = filter(data_array_in[2*tid+1].key);


        // -- PRESCAN -- //
        // Upsweep phase
        int spread = 1;
        for (int i = THREADS_PER_BLOCK; i > 0; i >>= 1)
        {
            __syncthreads();

            if (threadIdx.x < i)
            {
                int idx_first  = ((2*threadIdx.x+1) * spread) - 1;
                int idx_second = ((2*threadIdx.x+2) * spread) - 1;
                cache[idx_second] += cache[idx_first];
            }

            spread <<= 1; // Multiply by 2
        }

        // Set last element to 0 before the downsweep phase
        if (tid == 0) cache[2*THREADS_PER_BLOCK-1] = 0;

        // Downsweep phase
        spread >>= 1;
        for (int i = 1; i <= THREADS_PER_BLOCK; i <<= 1)
        {
            __syncthreads();

            if (threadIdx.x < i)
            {
                int idx_first  = ((2*threadIdx.x+1) * spread) - 1;
                int idx_second = ((2*threadIdx.x+2) * spread) - 1;

                int tmp = cache[idx_second];

                cache[idx_second] += cache[idx_first];  // Set the right child to L+current
                cache[idx_first]   = tmp;               // Set the left child to current
            }

            spread >>= 1; // Divide by 2
        }

        __syncthreads();


        data_prescan_out[2*tid]   = cache[2*threadIdx.x];
        data_prescan_out[2*tid+1] = cache[2*threadIdx.x+1];
    }

}


DataArray filterArray (const DataArray &da)
{
    std::cout << "Filtering data with CUDA!!" << std::endl;

    // Copy data to gpu
    Data* g_data_array_in;
    cudaMalloc((void**)&g_data_array_in, da.size*sizeof(Data));
    cudaMemcpy(g_data_array_in, da.array, da.size*sizeof(Data), cudaMemcpyHostToDevice);
    // Output data vector
    int* g_prescan_out;
    cudaMalloc((void**)&g_prescan_out, da.size*sizeof(int));

    // We use only one dimensional indices of grid cells and blocks because it is easier - we have a linear
    // vector of data
    // Each block can process double the amount of data than the number of threads in it
    int num_blocks = std::ceil(da.size / (2.0*THREADS_PER_BLOCK));

    prescan<<<num_blocks, THREADS_PER_BLOCK>>>(g_data_array_in, da.size, g_prescan_out);

    int prescan_out[da.size];
    cudaMemcpy(prescan_out, g_prescan_out, da.size*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(g_data_array_in);
    cudaFree(g_prescan_out);


    for (int i = 0; i < da.size; ++i)
    {
        std::cout << da.array[i].key << ": " << ((da.array[i].key >= FILTER_MIN && da.array[i].key <= FILTER_MAX) ? 1 : 0) << " " << prescan_out[i] << std::endl;
    }


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


