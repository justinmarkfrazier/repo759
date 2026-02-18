#include "stencil.cuh"
#include <cuda.h>
#include <iostream>
#include <cstddef>
#include <vector>
#include <random>
#include <chrono>

// Empty kernel to prevent cold start from affecting timing
__global__ void noop() {}


float rand_num(float min, float max) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

int main(int argc, char *argv[]) {
    
    // Read command line arguments
    const unsigned int n = std::atoi(argv[1]);
    const unsigned int R = std::atoi(argv[2]);
    const unsigned int threads_per_block = std::atoi(argv[3]);

    // Initialize image and mask with random values, initialize output with zeros
    std::vector<float> image(n);
    std::vector<float> mask(2*R+1);
    for (unsigned int i = 0; i < n; ++i) {
        image[i] = rand_num(-1.0f, 1.0f);
    }
    for (unsigned int i = 0; i < 2*R+1; ++i) {
        mask[i] = rand_num(-1.0f, 1.0f);
    }

    std::vector<float> output(n, 0.0f);

    // Timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Prevent cold start from affecting timing
    cudaFree(0);
    noop<<<1, 1>>>();
    cudaDeviceSynchronize();


    // Allocate device memory and zero out all entries in these device arrays
    float *d_image, *d_mask, *d_output;
    cudaMalloc((void**)&d_image, n * sizeof(float));
    cudaMalloc((void**)&d_mask, (2*R+1) * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));
    cudaMemset(d_image, 0, n * sizeof(float));
    cudaMemset(d_mask, 0, (2*R+1) * sizeof(float));
    cudaMemset(d_output, 0, n * sizeof(float));

    // Copy input data to device memory
    cudaMemcpy(d_image, image.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.data(), (2*R+1) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // time kernel execution
    cudaEventRecord(start);
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy output data back to host memory
    cudaMemcpy(output.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print last element of output and time taken
    std::cout << output[n - 1] << "\n";
    std::cout << milliseconds << "\n";

    // Free device memory
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);

    return 0;
}