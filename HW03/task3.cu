#include "vscale.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cuda.h>

// Empty kernel to prevent cold start from affecting timing
__global__ void noop() {}


float rand_num(float min, float max) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

int main(int argc, char *argv[]) {
    // Timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Prevent cold start from affecting timing
    cudaFree(0);
    noop<<<1, 1>>>();
    cudaDeviceSynchronize();


    int n = std::atoi(argv[1]);

    // Initialize host arrays with random values
    std::vector<float> hA(n), hB(n);
    for (int i = 0; i < n; ++i) {
        hA[i] = rand_num(-10.0f, 10.0f);
        hB[i] = rand_num(0.0f, 1.0f);
    }

    // Allocate device memory and zero out all entries in these device arrays
    float *dA, *dB;
    cudaMalloc((void**)&dA, n * sizeof(float));
    cudaMalloc((void**)&dB, n * sizeof(float));
    cudaMemset(dA, 0, n * sizeof(float));
    cudaMemset(dB, 0, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Time kernel execution
    cudaEventRecord(start);
    vscale<<<numBlocks, blockSize>>>(dA, dB, n);
    cudaEventRecord(stop);

    // Copy results back to host
    cudaMemcpy(hB.data(), dB, n * sizeof(float), cudaMemcpyDeviceToHost);
    // Synchronize before continuing
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print time, first element, and last element of the resulting array
    std::cout << milliseconds << "\n";
    std::cout << hB[0] << "\n";
    std::cout << hB.back() << "\n";

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);

    return 0;
}