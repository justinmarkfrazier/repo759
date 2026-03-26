#include "montecarlo.h"
#include <iostream>
#include <cstddef>
#include <algorithm>
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
    const unsigned int t = std::atoi(argv[2]);

    // rand num range
    const float r = 1.0f;
    const float i_min = -r, i_max = r;

    std::vector<float> x(n);
    std::vector<float> y(n);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] = rand_num<float>(i_min, i_max);
        y[i] = rand_num<float>(i_min, i_max);
    }

    omp_set_num_threads(t);

    double start = omp_get_wtime();
    int incircle = montecarlo(n, x.data(), y.data(), r);
    double end = omp_get_wtime();
    double duration_msec = (end - start) * 1000.0;
    
    float pi_est = 4.0f * static_cast<float>(incircle) / static_cast<float>(n);

    std::cout << pi_est << "\n";
    std::cout << duration_msec << "\n";

    return 0;
}