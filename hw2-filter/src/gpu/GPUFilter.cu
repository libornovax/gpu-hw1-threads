#include "GPUFilter.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


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


}


