#include "reduce.cuh"
#include <cuda.h>
#include <iostream>
#include <cstddef>
#include <vector>
#include <random>
#include <chrono>
#include <type_traits>
#include <cstdlib>

// Empty kernel to prevent cold start from affecting timing
__global__ void noop() {}

// one rand_num that works for int / float / double
template <typename T>
T rand_num(T min, T max) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> dist(min, max);
        return dist(rng);
    } else {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(rng);
    }
}

int main(int argc, char *argv[]) {

    // Read command line arguments
    const unsigned int n = std::atoi(argv[1]);
    const unsigned int threads_per_block = std::atoi(argv[2]);

    // random number range
    const float  f_min = -1.0f, f_max = 1.0f;

    // initialize and fillarray with random nums in range
    std::vector<float> A_vec(n);
    for (unsigned int i = 0; i < n; ++i) {
        A_vec[i] = rand_num<float>(f_min, f_max);
    }

    // Timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Prevent cold start from affecting timing
    cudaFree(0);
    noop<<<1, 1>>>();
    cudaDeviceSynchronize();

    // allocate memory
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMemcpy(d_in, A_vec.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    unsigned int elems_per_block = 2 * threads_per_block;
    unsigned int blocks = (n + elems_per_block - 1) / elems_per_block;
    cudaMalloc(&d_out, blocks * sizeof(float));

    // launch and time kernel
    cudaEventRecord(start);
    reduce(&d_in, &d_out, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // copy back output
    float sum = 0.0f;
    cudaMemcpy(&sum, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    // print output
    std::cout << sum << "\n";
    std::cout << ms << "\n";

    // free memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}