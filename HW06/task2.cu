#include "scan.cuh"
#include <cuda.h>
#include <iostream>
#include <cstddef>
#include <vector>
#include <random>
#include <chrono>
#include <type_traits>
#include <cstdlib>

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

int main(int argc, char *argv[])
{

    // Read command line arguments
    const unsigned int n = std::atoi(argv[1]);
    const unsigned int threads_per_block = std::atoi(argv[2]);

    // rand num range
    const float  f_min = -1.0f, f_max = 1.0f;

    // create and fill matrices A and B
    std::vector<float> arr_vec(n);

    for (unsigned int i = 0; i < n; ++i) {
        arr_vec[i] = rand_num<float>(f_min, f_max);
    }

    // Timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // allocate managed memory, copy vector values in, and fill C with zeros
    float  *arr, *output;

    cudaMallocManaged(&arr, n * sizeof(float));
    cudaMallocManaged(&output, n * sizeof(float));

    for (unsigned int i = 0; i < n; ++i) {
        arr[i] = arr_vec[i];
    }

    // Time the scan function
    cudaEventRecord(start);
    scan(arr, output, n, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    // print last element of output and time taken
    std::cout << output[n - 1] << "\n";
    std::cout << ms << "\n";

    // Cleanup
    cudaFree(arr);
    cudaFree(output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}