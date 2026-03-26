#include "cluster.h"
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

    // rand num range
    const float i_min = 0.0f, i_max = static_cast<float>(n);

    std::vector<float> arr(n);

    for (std::size_t i = 0; i < n; ++i) {
        arr[i] = rand_num<float>(i_min, i_max);
    }

    std::sort(arr.begin(), arr.end());

    std::vector<float> centers(t);

    for (std::size_t i = 0; i < t; ++i) {
        centers[i] = static_cast<float>(n) * (2.0f * i + 1.0f) / (2.0f * t);
    }

    std::vector<float> dists(t) = 0.0f;

    omp_set_num_threads(t);

    double start = omp_get_wtime();
    cluster(n, t, arr.data(), centers.data(), dists.data());
    double end = omp_get_wtime();
    double duration_msec = (end - start) * 1000.0;
    
    std::size_t max_id = 0;
    for (std::size_t i = 1; i < t; ++i) {
        if (dists[i] > dists[max_id]) {
            max_id = i;
        }
    }

    std::cout << dists[max_id] << "\n";
    std::cout << max_id << "\n";
    std::cout << duration_msec << "\n";

    return 0;
}