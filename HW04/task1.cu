#include "matmul.cuh"
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
    
    const unsigned int n = std::atoi(argv[1]);
    const unsigned int threads_per_block = std::atoi(argv[2]);

    // Initialize input matrices A and B with random values, initialize C with zeros
    std::vector<float> A_vec(n*n);
    std::vector<float> B_vec(n*n);
    for (unsigned int i = 0; i < n * n; ++i) {
        A_vec[i] = rand_num(-1.0f, 1.0f);
        B_vec[i] = rand_num(-1.0f, 1.0f);
    }

    std::vector<float> C(n*n, 0.0f);

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
    float *dA, *dB, *dC;
    cudaMalloc((void**)&dA, n * n * sizeof(float));
    cudaMalloc((void**)&dB, n * n * sizeof(float));
    cudaMalloc((void**)&dC, n * n * sizeof(float));
    cudaMemset(dA, 0, n * n * sizeof(float));
    cudaMemset(dB, 0, n * n * sizeof(float));
    cudaMemset(dC, 0, n * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(dA, A_vec.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B_vec.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Time kernel execution
    cudaEventRecord(start);
    matmul(dA, dB, dC, n, threads_per_block);
    cudaEventRecord(stop);

    // Synchronize before continuing
    cudaEventSynchronize(stop);

    // Copy result from device to host
    cudaMemcpy(C.data(), dC, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //Print last element of C and time taken
    std::cout << C[n*n - 1] << "\n";
    std::cout << milliseconds << "\n";

    return 0;
}