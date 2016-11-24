#include "GPUFilter.h"

#include <iostream>
#include <vector>
#include "settings.h"
#include "check_error.h"
#include "data.h"
#include "kernels.h"


namespace GPUFilter {

namespace {

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

        // Copy data to GPU
        Data* g_data_array_in;
        cudaMalloc((void**)&g_data_array_in, da.size*sizeof(Data));
        cudaMemcpy(g_data_array_in, da.array, da.size*sizeof(Data), cudaMemcpyHostToDevice);

        int* g_block_sums_out;
        cudaMalloc((void**)&g_block_sums_out, num_blocks*sizeof(int));

        filterPrescan<<<num_blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK>>>(g_data_array_in, da.size,
                                                                              g_indices_out, g_block_sums_out);

        g_index_pyramid.push_back(g_indices_out);
        g_index_pyramid.push_back(g_block_sums_out);

        if (num_blocks > 1)
        {
            // Call the recursive function
            determineIndicesRecursive(g_index_pyramid, 1, num_blocks);
        }


        cudaMemcpy(&output_size_out, g_index_pyramid[g_index_pyramid.size()-1], sizeof(int), cudaMemcpyDeviceToHost);


        int level_size = 2*THREADS_PER_BLOCK;
        for (int l = g_index_pyramid.size()-2; l > 0; --l)
        {
            propagateSum<<<2*level_size, THREADS_PER_BLOCK>>>(g_index_pyramid[l], g_index_pyramid[l-1],
                                                              level_size);

            level_size = level_size * 2*THREADS_PER_BLOCK;

            if (l != 0) cudaFree(g_index_pyramid[l]);
        }
    }

}


DataArray filterArray (const DataArray &da)
{
    std::cout << "Filtering data with CUDA!!" << std::endl;



    // Compute indices of elements in the output array - scan
    int* g_indices_out;
    cudaMalloc((void**)&g_indices_out, da.size*sizeof(int));
    int output_size;
    determineIndices(da, g_indices_out, output_size);

    // Copy data to GPU
    Data* g_data_array_in;
    cudaMalloc((void**)&g_data_array_in, da.size*sizeof(Data));
    cudaMemcpy(g_data_array_in, da.array, da.size*sizeof(Data), cudaMemcpyHostToDevice);

    // Copy data to the output array
    Data* g_da;
    cudaMalloc((void**)&g_da, output_size*sizeof(Data));

    int num_blocks = std::ceil(da.size / (2.0*THREADS_PER_BLOCK));
    copyElementsToOutput<<<2*num_blocks, THREADS_PER_BLOCK>>>(g_data_array_in, g_indices_out, g_da);


    DataArray out(output_size);
    cudaMemcpy(out.array, g_da, output_size*sizeof(Data), cudaMemcpyDeviceToHost);


    return out;
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

