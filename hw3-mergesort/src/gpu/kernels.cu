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
void bitonicSort (float *g_sequence_in_out, int length)
{
    // The ID of the thread - we use only one dimensional grid and blocks
    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    int s_sequence_length = 2 * blockDim.x;
    extern __shared__ float s_sequence[];  // Will be double the dimension of the block

    // Copy the input array into the shared memory (each thread copies 2 elements)
    s_sequence[2*threadIdx.x]   = g_sequence_in_out[2*tid];
    s_sequence[2*threadIdx.x+1] = g_sequence_in_out[2*tid+1];


    int x = threadIdx.x;
    for (int i = 2; i <= s_sequence_length; i <<= 1)
    {
        for (int j = i; j >= 2; j >> 1)
        {
            int j_half = j >> 1;
            int id1 = j * int(x/j_half) + x % j_half;
            int id2 = id1 + j_half;

            // Swap if out of order
            bool smaller1 = int(x/j_half) % 2 == 0;
            if ((s_sequence[id1] > s_sequence[id2]) == smaller1)
            {
                // Swap the numbers
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


