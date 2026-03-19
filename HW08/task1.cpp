#include "matmul.h"
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

using std::chrono::high_resolution_clock;
using std::chrono::duration;

int main(int argc, char *argv[])
{
    // Read command line arguments
    const std::size_t n = std::atoi(argv[1]);
    const unsigned int t = std::atoi(argv[2]);

    // rand num range
    const float  f_min = 0.0f, f_max = 1.0f;

    // create and fill matrices A and B
    std::vector<float> A(n*n);
    std::vector<float> B(n*n);
    std::vector<float> C(n * n, 0.0f);

    for (unsigned int i = 0; i < n * n; ++i) {
        A[i] = rand_num<float>(f_min, f_max);
        B[i] = rand_num<float>(f_min, f_max);
    }

    omp_set_num_threads(t);

    double start = omp_get_wtime();
    mmul(A.data(), B.data(), C.data(), n);
    double end = omp_get_wtime();
    double duration_msec = (end - start) * 1000.0;

    std::cout << C[0];
    std::cout << "\n" << C.back();
    std::cout << "\n" << duration_msec << "\n";

    return 0;
}