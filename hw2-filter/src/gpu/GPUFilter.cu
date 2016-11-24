#include "GPUFilter.h"

#include <iostream>
#include <vector>
#include "settings.h"
#include "check_error.h"
#include "data.h"
#include "kernels.h"


namespace GPUFilter {

namespace {

    void determineIndicesRecursive (std::vector<int*> &g_index_pyramid, std::vector<int> &level_sizes,
                                    int level, int level_size)
    {
        int num_blocks = std::ceil(level_size / (2.0*THREADS_PER_BLOCK));

        int* g_block_sums_out;
        cudaMalloc((void**)&g_block_sums_out, num_blocks*sizeof(int));

        onlyPrescan<<<num_blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK>>>(g_index_pyramid[level],
                                                                            level_size,
                                                                            g_index_pyramid[level],
                                                                            g_block_sums_out);

        g_index_pyramid.push_back(g_block_sums_out);
        level_sizes.push_back(num_blocks);

        if (num_blocks > 1)
        {
            // Call the recursive function
            determineIndicesRecursive(g_index_pyramid, level_sizes, level+1, num_blocks);
        }
    }


    void firstPyramidLevel (const DataArray &da, int *g_indices_out, std::vector<int*> &g_index_pyramid,
                            std::vector<int> &level_sizes)
    {
        // Each block can process double the amount of data than the number of threads in it
        int num_blocks = std::ceil(da.size / (2.0*THREADS_PER_BLOCK));

        // Copy data to GPU
        Data* g_data_array_in;
        cudaMalloc((void**)&g_data_array_in, da.size*sizeof(Data));
        cudaMemcpy(g_data_array_in, da.array, da.size*sizeof(Data), cudaMemcpyHostToDevice);

        // Array for block sums of the first level
        int* g_block_sums_out;
        cudaMalloc((void**)&g_block_sums_out, num_blocks*sizeof(int));


        // We need to first call kernel with filter function, which filters the elements of the input
        // array - marks the ones we want to keep. Then, prescan on the 0/1 membership array determines
        // indices within each block
        filterPrescan<<<num_blocks, THREADS_PER_BLOCK, 2*THREADS_PER_BLOCK>>>(g_data_array_in, da.size,
                                                                              g_indices_out,
                                                                              g_block_sums_out);


        // Store the results in the pyramid
        g_index_pyramid.push_back(g_indices_out); level_sizes.push_back(da.size);
        g_index_pyramid.push_back(g_block_sums_out); level_sizes.push_back(num_blocks);
    }


    /**
     * @brief Computes indices of the filtered elements (using prescan)
     * @param da Input array of Data structures
     * @param g_indices_out Output, array of size da.size with filtered element indices
     * @return Total number of filtered elements
     */
    int determineIndices (const DataArray &da, int *g_indices_out)
    {
        // The indices will be determined by prescan. The prescan must be parallel on the GPU and recursive
        // on CPU - if the output of the prescan cannot fit in one block then we have to recursively call
        // prescan on the new array
        //
        // The skeleton of this algorithm is taken from the prescan for arrays of arbitrary sizes in CUDA
        // samples: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/Projects/scanLargeArray.tar.gz
        //

        // Pyramid of partial results that will be created by calling recursive prescans (kept on a GPU)
        std::vector<int*> g_index_pyramid;
        std::vector<int>  level_sizes;  // Because we want to support arrays of arbitrary sizes - store them


        firstPyramidLevel(da, g_indices_out, g_index_pyramid, level_sizes);

        if (level_sizes.back() > 1)
        {
            // Call the recursive function
            determineIndicesRecursive(g_index_pyramid, level_sizes, 1, level_sizes.back());
        }


        // The top of the pyramid contains the total number of filtered elements (the top level was not
        // processed by prescan because it has one element)
        int output_size;
        cudaMemcpy(&output_size, g_index_pyramid[g_index_pyramid.size()-1], sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(g_index_pyramid[g_index_pyramid.size()-1]);

        for (int l = g_index_pyramid.size()-2; l > 0; --l)
        {
            int nbl = std::ceil(double(level_sizes[l-1]) / THREADS_PER_BLOCK);
            propagateSum<<<nbl, THREADS_PER_BLOCK>>>(g_index_pyramid[l], g_index_pyramid[l-1], level_sizes[l-1]);

            if (l != 0) cudaFree(g_index_pyramid[l]);
        }

        return output_size;
    }

}


DataArray filterArray (const DataArray &da)
{
    // Compute indices of elements in the output array - scan
    int* g_indices_out;
    cudaMalloc((void**)&g_indices_out, da.size*sizeof(int));

    int output_size = determineIndices(da, g_indices_out);


    // Copy data to GPU
    Data* g_da_in; cudaMalloc((void**)&g_da_in, da.size*sizeof(Data));
    cudaMemcpy(g_da_in, da.array, da.size*sizeof(Data), cudaMemcpyHostToDevice);

    // Copy data to the output array
    Data* g_da_out; cudaMalloc((void**)&g_da_out, output_size*sizeof(Data));

    int num_blocks = std::ceil(double(da.size) / THREADS_PER_BLOCK);
    copyElementsToOutput<<<num_blocks, THREADS_PER_BLOCK>>>(g_da_in, da.size, g_indices_out, g_da_out);


    DataArray out(output_size);
    cudaMemcpy(out.array, g_da_out, output_size*sizeof(Data), cudaMemcpyDeviceToHost);

    cudaFree(g_da_in);
    cudaFree(g_indices_out);

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

