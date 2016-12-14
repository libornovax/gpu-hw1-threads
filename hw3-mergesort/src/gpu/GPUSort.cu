#include "GPUSort.h"

#include <iostream>
#include <vector>
#include <cassert>
#include "settings.h"
#include "check_error.h"
#include "kernels.h"


namespace GPUSort {

    void sortSequence (std::vector<float> &seq)
    {
        // Copy data to GPU
        float* g_seq_in_out; cudaMalloc((void**)&g_seq_in_out, seq.size()*sizeof(float));
        cudaMemcpy(g_seq_in_out, seq.data(), seq.size()*sizeof(float), cudaMemcpyHostToDevice);

        int num_blocks      = seq.size() / (THREADS_PER_BLOCK * 2);
        int shared_mem_size = 2*THREADS_PER_BLOCK * sizeof(float);

        assert(seq.size() == num_blocks*THREADS_PER_BLOCK*2);  // We only support power on 2 sequence sizes


        bitonicSort<<< num_blocks, THREADS_PER_BLOCK, shared_mem_size >>>(g_seq_in_out, seq.size());


        cudaMemcpy(seq.data(), g_seq_in_out, seq.size()*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(g_seq_in_out);
    }


    bool initialize ()
    {
        // Find out if there is a CUDA capable device
        int device_count;
        CHECK_ERROR(cudaGetDeviceCount(&device_count));

        // Get properties of the device
        cudaDeviceProp device_properties;
        CHECK_ERROR(cudaGetDeviceProperties(&device_properties, 0));

        if (device_count == 0 || (device_properties.major == 0 && device_properties.minor == 0))
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

