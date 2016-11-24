//
// Libor Novak
// 11/21/2016
//
// GPU class, HW2
// Implementation of an array filter with CUDA
//

#include <iostream>
#include <chrono>
#include "settings.h"
#include "ArrayGenerator.h"
#include "cpu/CPUFilter.h"
#include "gpu/GPUFilter.h"


int main (int argc, char* argv[])
{
    std::cout << "-- ARRAY SIZE: " << ARRAY_SIZE << " (" << (double(ARRAY_SIZE*sizeof(Data))/(1024*1024)) << " MB)" << std::endl;
    std::cout << "-- BLOCK SIZE: " << THREADS_PER_BLOCK << " threads" << std::endl;

    DataArray da = ArrayGenerator::generateRandomArray(ARRAY_SIZE);

#ifdef MEASURE_TIME
    auto start1 = std::chrono::high_resolution_clock::now();
#endif
    // Filter data on CPU
    DataArray da_cpu_filtered = CPUFilter::filterArray(da);
#ifdef MEASURE_TIME
    auto end1 = std::chrono::high_resolution_clock::now();
#endif


    if (GPUFilter::initialize())
    {
#ifdef MEASURE_TIME
        auto start2 = std::chrono::high_resolution_clock::now();
#endif
        // Filter data on GPU
        DataArray da_gpu_filtered = GPUFilter::filterArray(da);
#ifdef MEASURE_TIME
        auto end2 = std::chrono::high_resolution_clock::now();
        std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count() << " ms" << std::endl;
        std::cout << "GPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count() << " ms" << std::endl;
#endif

        // Check if the CPU and GPU output equals
        if (da_cpu_filtered.size != da_gpu_filtered.size)
        {
            std::cout << "ERROR: Length of CPU (" << da_cpu_filtered.size << ") and GPU (" << da_gpu_filtered.size << ") filtered data are not the same." << std::endl;
        }
        int num_errors = 0;
        for (size_t i = 0; i < da_cpu_filtered.size; ++i)
        {
            if (da_cpu_filtered.array[i].key != da_gpu_filtered.array[i].key ||
                    da_cpu_filtered.array[i].data != da_gpu_filtered.array[i].data)
            {
                std::cout << "ERROR: Data entry from CPU (" << da_cpu_filtered.array[i].key << ": " << da_cpu_filtered.array[i].data << ") and GPU (" << da_gpu_filtered.array[i].key << ": " << da_gpu_filtered.array[i].data << ") data is not the same." << std::endl;
                num_errors++;
            }
        }
        if (num_errors == 0) std::cout << "-- CONGRATULATIONS! GPU solution is correct!" << std::endl;
    }
    else
    {
#ifdef MEASURE_TIME
        std::cout << "CPU time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1-start1).count() << " ms" << std::endl;
#endif
    }


    return EXIT_SUCCESS;
}
