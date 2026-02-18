#include <cuda.h>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R) {
    
    // dynamic shared memory
    extern __shared__ float sh[];                    // this will get divided into three parts: mask, image tile, and output tile

    // setup shared memory pointers
    unsigned int mask_len = 2 * R + 1;               // this is assumed
    unsigned int tile_len = blockDim.x + 2 * R;      // this is the length of the tile that the whole block will process, so the center elements plus the left and right outer mask regions

    float* s_mask = sh;                              // mask is at the beginning of shared memory (same pointer as sh)
    float* s_image = s_mask + mask_len;              // image tile is next
    float* s_output = s_image + tile_len;            // output tile is last

    // compute thread IDs
    unsigned int thread_id = threadIdx.x;
    unsigned int global_id = blockIdx.x * blockDim.x + thread_id;

    // load entire mask into shared memory
    for (unsigned int i = thread_id; i < mask_len; i += blockDim.x) {
        s_mask[i] = mask[i];
    }

    /////////// load image elements needed by the block into shared memory
    /////////// each element needs image[i-R] to image[i+R]
    /////////// each block computes output element global_id from (blockIdx.x * blockDim.x) to (blockIdx.x * blockDim.x + blockDim.x - 1)

    // each thread loads center element image[global_id] to shared memory s_image[thread_id + R]
    if (global_id < n) {
        s_image[thread_id + R] = image[global_id];
    } 
    else {
        s_image[thread_id + R] = 1.0f; // pad if out of bounds
    }

    // load left halo elements (load to s_image[thread_id])
    // these can be less than zero but not greater than n-1, only worry about zero
    if (thread_id < R) {
        int left_index = global_id - R;
        if (left_index >= 0) {
            s_image[thread_id] = image[left_index];
        } 
        else {
            s_image[thread_id] = 1.0f; // pad if out of bounds
        }
    }

    //load right halo elements (load to s_image[thread_id + 2 * R])
    if (thread_id >= blockDim.x - R) {
        unsigned int right_index = global_id + R;
        if (right_index < n) {
            s_image[thread_id + 2 * R] = image[right_index];
        } 
        else {
            s_image[thread_id + 2 * R] = 1.0f; // pad if out of bounds
        }
    }

    __syncthreads();

    // compute output element
    if (global_id < n) {
        float sum = 0.0f;
        for (int j = -(int)R; j <= (int)R; j++) {
            sum += s_image[thread_id + j + R] * s_mask[j + R];
        }
        s_output[thread_id] = sum;
    }

    __syncthreads();

    // write output back to global memory
    if (global_id < n) {
        output[global_id] = s_output[thread_id];
    }
}


__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block) {

    // we know threads_per_block is >= 2R+1
    dim3 blockDim(threads_per_block);
    dim3 gridDim((n + threads_per_block - 1) / threads_per_block);

    // calculate shared memory size: mask + image tile + output tile
    unsigned int mask_len = 2 * R + 1;
    unsigned int tile_len = threads_per_block + 2 * R;
    size_t shared_mem_size = (mask_len + tile_len + blockDim.x) * sizeof(float);

    // launch kernel
    stencil_kernel<<<gridDim, blockDim, shared_mem_size>>>(image, mask, output, n, R);
}