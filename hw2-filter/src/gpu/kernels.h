#ifndef KERNELS_H
#define KERNELS_H

#include "data.h"
#include <cuda.h>
#include <cuda_runtime.h>


__global__
void filterPrescan (Data *g_data_array_in, int length, int *g_prescan_out, int *g_block_sums_out);

__global__
void onlyPrescan (int *g_array_in, int length, int *g_prescan_out, int *g_block_sums_out);

__global__
void propagateSum (int *g_level_top, int *g_level_bottom, int top_level_size);

__global__
void copyElementsToOutput (Data *g_da, int *g_indices, Data *g_da_out);


#endif // KERNELS_H

