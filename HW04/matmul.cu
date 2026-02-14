#include "matmul.cuh"
#include <cuda.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {

    // Compute the row and column index of the element this thread should compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Include index guard
    if (row < n && col < n) {
        float value = 0;
        for (int k = 0; k < n; ++k) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Calculate grid and block dimensions
    dim3 blockDim(32, threads_per_block / 32);
    dim3 gridDim(
        (unsigned int)((n + blockDim.x - 1) / blockDim.x),
        (unsigned int)((n + blockDim.y - 1) / blockDim.y)
    );

    // Launch the kernel
    matmul_kernel<<<gridDim, blockDim>>>(A, B, C, n);
}