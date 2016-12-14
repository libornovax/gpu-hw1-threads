#include "GPUSort.h"

#include <iostream>
#include <vector>
#include <cassert>
#include "settings.h"
#include "check_error.h"
#include "kernels.h"


namespace GPUSort {

    void bitonicSortHost (float *g_seq_in_out, int length)
    {
        // Number of blocks we have to launch to process the whole sequence - each block processes 2 times
        // its size of elements
        int num_blocks = length / (2*THREADS_PER_BLOCK);
        int shared_mem_size = std::min(2*THREADS_PER_BLOCK, length) * sizeof(float);

        assert(length == num_blocks*THREADS_PER_BLOCK*2);  // We only support 2^n sequence sizes


        for (int i = 2; i <= length; i <<= 1)
        {
            for (int j = i; j >= 2; j >>= 1)
            {
                if (j <= 2*THREADS_PER_BLOCK)
                {
                    // We can process the rest in parallel without interfering (we do not need to synchronize
                    // the blocks at this level)
                    bitonicSort<<< num_blocks, THREADS_PER_BLOCK, shared_mem_size >>>(g_seq_in_out, length);
                    break;
                }
                else
                {
                    // The spread of the comparators is too big to be processed without block synchronization
                    // Launch only this comparison on a GPU
                    bitonicCompare<<< num_blocks, THREADS_PER_BLOCK >>>(g_seq_in_out, length, i, j);
                }

                cudaDeviceSynchronize();
            }
        }
    }


    void sortSequence (std::vector<float> &seq)
    {
        // Copy data to GPU
        float* g_seq_in_out; cudaMalloc((void**)&g_seq_in_out, seq.size()*sizeof(float));
        cudaMemcpy(g_seq_in_out, seq.data(), seq.size()*sizeof(float), cudaMemcpyHostToDevice);


        bitonicSortHost(g_seq_in_out, seq.size());


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

