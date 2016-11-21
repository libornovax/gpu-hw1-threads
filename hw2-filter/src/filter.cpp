//
// Libor Novak
// 11/21/2016
//
// GPU class, HW2
// Implementation of an array filter with CUDA
//

#include <iostream>
#include "settings.h"
#include "ArrayGenerator.h"
#include "cpu/CPUFilter.h"


int main (int argc, char* argv[])
{
    DataArray da = ArrayGenerator::generateRandomArray(ARRAY_SIZE);

    // Filter data on CPU
    DataArray da_cpu_filtered = CPUFilter::filterArray(da);

    // Filter data on GPU
//    DataArray da_gpu_filtered = GPUFilter::filterArray(da);


    for (size_t i = 0; i < da_cpu_filtered.size; ++i)
    {
        std::cout << da_cpu_filtered.array[i].key << ": " << da_cpu_filtered.array[i].data << std::endl;
    }


    // Check if the CPU and GPU output equals
//    if (da_cpu_filtered.size != da_gpu_filtered.size)
//    {
//        std::cout << "ERROR: Length of CPU (" << da_cpu_filtered.size << ") and GPU (" << da_gpu_filtered.size << ") filtered data are not the same." << std::endl;
//    }
//    for (size_t i = 0; i < da_cpu_filtered.size; ++i)
//    {
//        if (da_cpu_filtered.array[i].key != da_gpu_filtered.array[i].key ||
//                da_cpu_filtered.array[i].data != da_gpu_filtered.array[i].data)
//        {
//            std::cout << "ERROR: Data entry from CPU (" << da_cpu_filtered.array[i].key << ": " << da_cpu_filtered.array[i].data << ") and GPU (" << da_gpu_filtered.array[i].key << ": " << da_gpu_filtered.array[i].data << ") data is not the same." << std::endl;
//        }
//    }


    return EXIT_SUCCESS;
}
