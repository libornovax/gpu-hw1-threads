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
//        return key >= FILTER_MIN && key <= FILTER_MAX;
        return key-FILTER_MIN <= FILTER_MAX-FILTER_MIN;
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
        std::cout << da.array[i].key << ": " << prescan_out[i] << std::endl;
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


