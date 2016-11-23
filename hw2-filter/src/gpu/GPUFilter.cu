#include "GPUFilter.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "check_error.h"


namespace GPUFilter {

namespace {

    /**
     * @brief Returns true if the key is in the interval, false if it should be thrown away
     * @param key
     * @return
     */
//    __device__ bool filter (int key)
//    {
//        return true;
//    }


    __global__ void addNumbers (int a, int b, int *result)
    {
        *result = a + b;
    }

}


DataArray filterArray (const DataArray &da)
{
    std::cout << "Filtering data with CUDA!!" << std::endl;

    int result;
    int* gpu_result;

    cudaMalloc((void**)&gpu_result, sizeof(int));

    addNumbers<<<1, 1>>>(10, 15, gpu_result);

    cudaMemcpy(&result, gpu_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpu_result);

    std::cout << "CUDA Result: " << result << std::endl;

    return DataArray(10);
}


bool initialize ()
{
    // Just malloc a dummy in the GPU memory
    int* gpu_dummy;
    cudaMalloc((void**)&gpu_dummy, sizeof(int));
    cudaFree(gpu_dummy);



    int device_count;
    CHECK_ERROR(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        // Error, we cannot initialize
        return false;
    }
    else
    {
        // Get properties of the device
        cudaDeviceProp device_properties;
        CHECK_ERROR(cudaGetDeviceProperties(&device_properties, 0));

        std::cout << "Device name:           " << device_properties.name << std::endl;
        std::cout << "Compute capability:    " << device_properties.major << "." << device_properties.minor << std::endl;
        std::cout << "Clock rate:            " << device_properties.clockRate << std::endl;
        std::cout << "Total global memory:   " << device_properties.totalGlobalMem << std::endl;
        std::cout << "Multiprocessor count:  " << device_properties.multiProcessorCount << std::endl;
        std::cout << "Max threads per block: " << device_properties.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads dim:       " << device_properties.maxThreadsDim << std::endl;

        return true;
    }
}


}


