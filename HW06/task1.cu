#include "mmul.h"
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
    const unsigned int n_tests = std::atoi(argv[2]);

    // rand num range
    const float  f_min = -1.0f, f_max = 1.0f;

    // create and fill matrices A and B
    std::vector<float> A_vec(n*n);
    std::vector<float> B_vec(n*n);

    for (unsigned int i = 0; i < n * n; ++i) {
        A_vec[i] = rand_num<float>(f_min, f_max);
        B_vec[i] = rand_num<float>(f_min, f_max);
    }

    // Timing variables
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_total = 0.0f;

    // allocate managed memory, copy vector values in, and fill C with zeros
    float  *A, *B, *C;

    cudaMallocManaged(&A, n * n * sizeof(float));
    cudaMallocManaged(&B, n * n * sizeof(float));
    cudaMallocManaged(&C, n * n * sizeof(float));

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = A_vec[i];  B[i] = B_vec[i];  C[i] = 0.0f;
    }

    // create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // warmup
    mmul(handle, A, B, C, n);

    //launch n_tests of mmul and record the whole loop
    cudaEventRecord(start);
    for (unsigned int t = 0; t < n_tests; ++t) {
        mmul(handle, A, B, C, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // calc and print time
    cudaEventElapsedTime(&ms_total, start, stop);
    float ms_avg = ms_total / n_tests;
    std::cout << ms_avg << "\n";

    //cleanup
    cudaFree(A); cudaFree(B); cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}