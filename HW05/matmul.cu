#include "matmul.cuh"
#include <cuda.h>

#define MAX_BLOCK_DIM 32  // because 32x32 = 1024 threads (max per block)

__global__ void matmul_kernel_1(const int* A, const int* B, int* C, unsigned int n)
{
    __shared__ int As[MAX_BLOCK_DIM][MAX_BLOCK_DIM];
    __shared__ int Bs[MAX_BLOCK_DIM][MAX_BLOCK_DIM];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int bd  = blockDim.x;                 // block_dim
    unsigned int row = blockIdx.y * bd + ty;
    unsigned int col = blockIdx.x * bd + tx;

    int acc = 0;

    unsigned int numTiles = (n + bd - 1) / bd;
    for (unsigned int t = 0; t < numTiles; ++t) {

        unsigned int a_col = t * bd + tx;          // k index for A
        unsigned int b_row = t * bd + ty;          // k index for B

        // Load tiles (pad with zeros if out of bounds)
        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0;

        __syncthreads();

        for (unsigned int k = 0; k < bd; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

__global__ void matmul_kernel_2(const float* A, const float* B, float* C, unsigned int n)
{
    __shared__ float As[MAX_BLOCK_DIM][MAX_BLOCK_DIM];
    __shared__ float Bs[MAX_BLOCK_DIM][MAX_BLOCK_DIM];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int bd  = blockDim.x;
    unsigned int row = blockIdx.y * bd + ty;
    unsigned int col = blockIdx.x * bd + tx;

    float acc = 0.0f;

    unsigned int numTiles = (n + bd - 1) / bd;
    for (unsigned int t = 0; t < numTiles; ++t) {

        unsigned int a_col = t * bd + tx;
        unsigned int b_row = t * bd + ty;

        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

        __syncthreads();

        for (unsigned int k = 0; k < bd; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

__global__ void matmul_kernel_3(const double* A, const double* B, double* C, unsigned int n)
{
    __shared__ double As[MAX_BLOCK_DIM][MAX_BLOCK_DIM];
    __shared__ double Bs[MAX_BLOCK_DIM][MAX_BLOCK_DIM];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    unsigned int bd  = blockDim.x;
    unsigned int row = blockIdx.y * bd + ty;
    unsigned int col = blockIdx.x * bd + tx;

    double acc = 0.0;

    unsigned int numTiles = (n + bd - 1) / bd;
    for (unsigned int t = 0; t < numTiles; ++t) {

        unsigned int a_col = t * bd + tx;
        unsigned int b_row = t * bd + ty;

        As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0;
        Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0;

        __syncthreads();

        for (unsigned int k = 0; k < bd; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = acc;
    }
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim)
{
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim,
                 (n + block_dim - 1) / block_dim);

    matmul_kernel_1<<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim)
{
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim,
                 (n + block_dim - 1) / block_dim);

    matmul_kernel_2<<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim)
{
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim,
                 (n + block_dim - 1) / block_dim);

    matmul_kernel_3<<<dimGrid, dimBlock>>>(A, B, C, n);
    cudaDeviceSynchronize();
}