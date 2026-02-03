#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include "matmul.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

float rand_num(float min, float max)
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

int main()
{
    // Timing variables
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_msec;

    const unsigned int n = 1000;
    std::cout << n;

    std::vector<double> A(n * n), B(n * n), C(n * n);

    for (std::size_t i = 0; i < A.size(); ++i) {
        A[i] = rand_num(0.0, 1.0);
        B[i] = rand_num(0.0, 1.0);
    }

    // Helper lambda to run and time each matrix multiplication
    auto run_raw = [&](auto fn) {
        std::fill(C.begin(), C.end(), 0.0);
        start = high_resolution_clock::now();
        fn();
        end = high_resolution_clock::now();
        duration_msec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
        std::cout << "\n" << duration_msec.count();
        std::cout << "\n" << C.back();
    };

    run_raw([&] { mmul1(A.data(), B.data(), C.data(), n); });
    run_raw([&] { mmul2(A.data(), B.data(), C.data(), n); });
    run_raw([&] { mmul3(A.data(), B.data(), C.data(), n); });
    run_raw([&] { mmul4(A, B, C.data(), n); });

    return 0;
}