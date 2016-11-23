#include "check_error.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


void HandleError(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf( "%s in %s at line %d\n", cudaGetErrorString( error ), file, line );
//        exit( EXIT_FAILURE );
    }
}

