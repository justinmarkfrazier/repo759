#include "matmul.cuh"
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
    const unsigned int block_dim = std::atoi(argv[2]);

    // pick ranges (edit these however you want)
    const int    i_min = -10,   i_max = 10;
    const float  f_min = -1.0f, f_max = 1.0f;
    const double d_min = -1.0,  d_max = 1.0;

    // --- int matrices (for matmul_1) ---
    std::vector<int> A1_vec(n*n);
    std::vector<int> B1_vec(n*n);
    std::vector<int> C1_vec(n*n, 0);

    for (unsigned int i = 0; i < n * n; ++i) {
        A1_vec[i] = rand_num<int>(i_min, i_max);
        B1_vec[i] = rand_num<int>(i_min, i_max);
    }

    // --- float matrices (for matmul_2) ---
    std::vector<float> A2_vec(n*n);
    std::vector<float> B2_vec(n*n);
    std::vector<float> C2_vec(n*n, 0.0f);

    for (unsigned int i = 0; i < n * n; ++i) {
        A2_vec[i] = rand_num<float>(f_min, f_max);
        B2_vec[i] = rand_num<float>(f_min, f_max);
    }

    // --- double matrices (for matmul_3) ---
    std::vector<double> A3_vec(n*n);
    std::vector<double> B3_vec(n*n);
    std::vector<double> C3_vec(n*n, 0.0);

    for (unsigned int i = 0; i < n * n; ++i) {
        A3_vec[i] = rand_num<double>(d_min, d_max);
        B3_vec[i] = rand_num<double>(d_min, d_max);
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

    // ---- allocate managed memory (matches matmul.cuh comment) ----
    int    *A1, *B1, *C1;
    float  *A2, *B2, *C2;
    double *A3, *B3, *C3;

    cudaMallocManaged(&A1, n * n * sizeof(int));
    cudaMallocManaged(&B1, n * n * sizeof(int));
    cudaMallocManaged(&C1, n * n * sizeof(int));

    cudaMallocManaged(&A2, n * n * sizeof(float));
    cudaMallocManaged(&B2, n * n * sizeof(float));
    cudaMallocManaged(&C2, n * n * sizeof(float));

    cudaMallocManaged(&A3, n * n * sizeof(double));
    cudaMallocManaged(&B3, n * n * sizeof(double));
    cudaMallocManaged(&C3, n * n * sizeof(double));

    // ---- copy vectors into managed arrays (and ensure C starts at 0) ----
    for (unsigned int i = 0; i < n * n; ++i) {
        A1[i] = A1_vec[i];  B1[i] = B1_vec[i];  C1[i] = 0;
        A2[i] = A2_vec[i];  B2[i] = B2_vec[i];  C2[i] = 0.0f;
        A3[i] = A3_vec[i];  B3[i] = B3_vec[i];  C3[i] = 0.0;
    }

    // ---- time matmul_1 (int) ----
    float ms1 = 0.0f;
    cudaEventRecord(start);
    matmul_1(A1, B1, C1, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms1, start, stop);

    std::cout << C1[0] << "\n";
    std::cout << C1[n*n - 1] << "\n";
    std::cout << ms1 << "\n";

    // ---- time matmul_2 (float) ----
    float ms2 = 0.0f;
    cudaEventRecord(start);
    matmul_2(A2, B2, C2, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms2, start, stop);

    std::cout << C2[0] << "\n";
    std::cout << C2[n*n - 1] << "\n";
    std::cout << ms2 << "\n";

    // ---- time matmul_3 (double) ----
    float ms3 = 0.0f;
    cudaEventRecord(start);
    matmul_3(A3, B3, C3, n, block_dim);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms3, start, stop);

    std::cout << C3[0] << "\n";
    std::cout << C3[n*n - 1] << "\n";
    std::cout << ms3 << "\n";

    // ---- cleanup ----
    cudaFree(A1); cudaFree(B1); cudaFree(C1);
    cudaFree(A2); cudaFree(B2); cudaFree(C2);
    cudaFree(A3); cudaFree(B3); cudaFree(C3);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}