#include "msort.h"
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
    const unsigned int t = std::atoi(argv[2]);
    const unsigned int ts = std::atoi(argv[3]);

    // rand num range
    const int i_min = -1000, i_max = 1000;

    std::vector<int> arr(n);

    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = rand_num<int>(i_min, i_max);
    }

    omp_set_num_threads(t);

    double start = omp_get_wtime();
    msort(arr.data(), n, ts);
    double end = omp_get_wtime();
    double duration_msec = (end - start) * 1000.0;
    
    std::cout << arr[0] << "\n";
    std::cout << arr[n - 1] << "\n";
    std::cout << duration_msec << "\n";

    return 0;
}