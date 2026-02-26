#include "reduce.cuh"
#include <cuda.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n)
{
    extern __shared__ float sdata[];

    // --- first add during load (kernel #4 idea from slides) ---
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // NOTE: bounds handling so N doesn't have to be a power of 2
    float x = 0.0f;
    if (i < n) x += g_idata[i];
    if (i + blockDim.x < n) x += g_idata[i + blockDim.x];

    sdata[tid] = x;
    __syncthreads();

    // --- reduce in shared memory (sequential addressing loop from slides) ---
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // --- write per-block result ---
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block)
{
    float *d_in  = *input;
    float *d_out = *output;

    unsigned int n = N;

    while (n > 1) {
        unsigned int elems_per_block = 2 * threads_per_block;  // kernel #4 loads 2 per thread
        unsigned int blocks = (n + elems_per_block - 1) / elems_per_block;

        size_t shmem = threads_per_block * sizeof(float);
        reduce_kernel<<<blocks, threads_per_block, shmem>>>(d_in, d_out, n);

        // next stage reduces the array of length = blocks
        n = blocks;

        // ping-pong buffers
        float *tmp = d_in;
        d_in = d_out;
        d_out = tmp;
    }

    // ensure result is written to (*input)[0]
    if (d_in != *input) {
        cudaMemcpy(*input, d_in, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    // required for timing correctness
    cudaDeviceSynchronize();
}