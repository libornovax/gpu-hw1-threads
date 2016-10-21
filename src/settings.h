#ifndef SETTINGS_H
#define SETTINGS_H


// Define the dimension of the matrix A of the linear equation system
#define MATRIX_DIM 10

// Number of matrices to be processed by the system
#define MATRIX_COUNT 50

// Size of the equation buffer on the input to the worker thread pool
#define STAGE2_BUFFER_SIZE 5

// Number of GEM worker threads
#define STAGE2_WORKERS_COUNT 3


#endif // SETTINGS_H

