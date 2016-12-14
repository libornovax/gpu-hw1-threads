#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>


__global__
void bitonicSort (float *g_sequence_in_out, int length);


#endif // KERNELS_H

