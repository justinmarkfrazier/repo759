#include "scan.cuh"
#include <cuda.h>

// Per-block Hillis–Steele inclusive scan.
// Also writes block sums if block_sums != nullptr.
__global__ void hillis_steele(const float* input,
                             float* output,
                             float* block_sums,
                             unsigned int n)
{
    extern __shared__ float temp[];  // size = 2 * blockDim.x floats

    unsigned int thid = threadIdx.x;
    unsigned int gid  = blockIdx.x * blockDim.x + thid;

    int pout = 0;
    int pin  = 1;

    // Inclusive scan: load the element itself (pad out-of-range with 0)
    temp[pout * blockDim.x + thid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (unsigned int offset = 1; offset < blockDim.x; offset *= 2) {
        pout = 1 - pout;
        pin  = 1 - pout;

        if (thid >= offset) {
            temp[pout * blockDim.x + thid] =
                temp[pin * blockDim.x + thid] +
                temp[pin * blockDim.x + thid - offset];
        } else {
            temp[pout * blockDim.x + thid] =
                temp[pin * blockDim.x + thid];
        }
        __syncthreads();
    }

    if (gid < n) {
        output[gid] = temp[pout * blockDim.x + thid];
    }

    // Block total is the last entry of the block's scanned shared array.
    // Padding with zeros makes this correct for partial last blocks.
    if (block_sums && thid == blockDim.x - 1) {
        block_sums[blockIdx.x] = temp[pout * blockDim.x + thid];
    }
}

// Adds scanned block offsets to each element.
// For block b>0, add scanned_block_sums[b-1].
__global__ void add_block_offsets(float* output,
                                 const float* scanned_block_sums,
                                 unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    float offset = 0.0f;
    if (blockIdx.x > 0) {
        offset = scanned_block_sums[blockIdx.x - 1];
    }
    output[gid] += offset;
}

__host__ void scan(const float* input, float* output,
                   unsigned int n, unsigned int threads_per_block)
{
    unsigned int threads = threads_per_block;
    unsigned int blocks  = (n + threads - 1) / threads;

    size_t shmem_bytes = 2 * (size_t)threads * sizeof(float);

    // One block covers the whole array
    if (blocks <= 1) {
        hillis_steele<<<1, threads, shmem_bytes>>>(input, output, nullptr, n);
        return;
    }

    // Because n <= threads_per_block * threads_per_block, we have blocks <= threads.
    // So scanning block_sums always fits in ONE block.

    float* block_sums = nullptr;
    cudaMallocManaged(&block_sums, (size_t)blocks * sizeof(float));

    // 1) Per-block scans + block sums
    hillis_steele<<<blocks, threads, shmem_bytes>>>(input, output, block_sums, n);

    // 2) Scan the block sums in-place (one block is enough)
    hillis_steele<<<1, threads, shmem_bytes>>>(block_sums, block_sums, nullptr, blocks);

    // 3) Add offsets to each block
    add_block_offsets<<<blocks, threads>>>(output, block_sums, n);

    // Note: freeing can force synchronization; leaving it here is simplest.
    cudaFree(block_sums);
}