#ifndef CHECK_ERROR_H
#define CHECK_ERROR_H


void HandleError(cudaError_t error, const char *file, int line);

#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )


#endif // CHECK_ERROR_H

