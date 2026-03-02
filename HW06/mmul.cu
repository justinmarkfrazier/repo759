#include "mmul.h"
#include <cuda.h>

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n)
{
    // we need to define alpha and beta for the function to work
    const float alpha = 1.0f;
    const float beta = 1.0f;

    // we call the cuBLAS function for matrix multiplication and sync for timing
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                A, n,
                B, n,
                &beta,
                C, n);
    
    cudaDeviceSynchronize();
}