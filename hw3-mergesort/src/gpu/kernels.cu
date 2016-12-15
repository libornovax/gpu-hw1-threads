#include "kernels.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "settings.h"


namespace {


}


// //////////////////////////////////////////////////////////////////////////////////////////////////////// //
// --------------------------------------------  CUDA KERNELS  -------------------------------------------- //
// //////////////////////////////////////////////////////////////////////////////////////////////////////// //

__global__
void bitonicSort (float *g_sequence_in_out, int length, int i_global)
{
    // The ID of the thread - we use only one dimensional grid and blocks
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    int s_sequence_length = (length < 2*blockDim.x) ? length : 2*blockDim.x;
    extern __shared__ float s_sequence[];  // Will be double the dimension of the block

    if (threadIdx.x < s_sequence_length/2)
    {
        // Copy the input array into the shared memory (each thread copies 2 elements)
        s_sequence[2*threadIdx.x]   = g_sequence_in_out[2*tid];
        s_sequence[2*threadIdx.x+1] = g_sequence_in_out[2*tid+1];

        // This variable says, whether the sequence should be sorted in reverse. Since this kernel is called
        // in an ensemble together with other blocks, we need to determine, in which order this block needs
        // to be sorted. It is basically the same as when we determine in which order we want to have
        // the elements (bool smaller1 = int(x / i_half) % 2 == 0;)
        bool reverse_order = int(blockIdx.x / (i_global / (2*blockDim.x))) % 2 == 0;


        // These two for cycles simulate the recursion of the sorting algorithm (see example Python code
        // in https://en.wikipedia.org/wiki/Bitonic_sorter). The first for cycle, which iterates over i is
        // the bitonic_sort nested recursion (top level just splitting the sequence to smaller sequences).
        // The for cycle iterating over j are the bitonic_merge recursion nested calls.
        //
        // Basically i says what is the depth of recursion that we are in the sorting (number of elements in
        // the comparator) and j is the depth of the merging recursion (number of elements in the merging
        // phase).

        int x = threadIdx.x;
        for (int i = 2; i <= s_sequence_length; i <<= 1)
        {
            // Just so we do not have to compute it again
            int i_half = i >> 1;

            for (int j = i; j >= 2; j >>= 1)
            {
                // Just so we do not have to compute it again
                int j_half = j >> 1;

                // Index of the first processed element
                int id1 = j * int(x/j_half) + x % j_half;
                // Index of the second processed element - note that the separation of the elements depends
                // on j (that is what the j variable is for)
                int id2 = id1 + j_half;

                // Determine in which order are the numbers supposed to be in this iteration. The smaller1
                // variable is true if the first element (id1) is supposed to be the smaller one
                bool smaller1 = int(x / i_half) % 2 == 0;
                if ((s_sequence[id1] > s_sequence[id2]) == !(smaller1 != reverse_order))
                {
                    // The numbers are in the wrong order -> swap them
                    float tmp = s_sequence[id1];
                    s_sequence[id1] = s_sequence[id2];
                    s_sequence[id2] = tmp;
                }

                __syncthreads();
            }
        }


        // Copy data to the output
        g_sequence_in_out[2*tid]   = s_sequence[2*threadIdx.x];
        g_sequence_in_out[2*tid+1] = s_sequence[2*threadIdx.x+1];
    }
}


__global__
void bitonicCompare (float *g_sequence_in_out, int length, int i, int j)
{
    // The ID of the thread - we use only one dimensional grid and blocks
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    int i_half = i >> 1;
    int j_half = j >> 1;

    // Index of the first processed element
    int id1 = j * int(x/j_half) + x % j_half;
    // Index of the second processed element - note that the separation of the elements depends
    // on j (that is what the j variable is for)
    int id2 = id1 + j_half;

    // Determine in which order are the numbers supposed to be in this iteration. The smaller1
    // variable is true if the first element (id1) is supposed to be the smaller one
    bool smaller1 = int(x / i_half) % 2 == 0;
    if ((g_sequence_in_out[id1] > g_sequence_in_out[id2]) == smaller1)
    {
        // The numbers are in the wrong order -> swap them
        float tmp = g_sequence_in_out[id1];
        g_sequence_in_out[id1] = g_sequence_in_out[id2];
        g_sequence_in_out[id2] = tmp;
    }
}


