#include "convolution.h"
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
    const unsigned int n = std::atoi(argv[1]);
    const unsigned int t = std::atoi(argv[2]);

    // rand num range
    const float  i_min = -10.0f, i_max = 10.0f;
    const float  m_min = -1.0f, m_max = 1.0f;

    std::vector<float> image(n * n);
    std::vector<float> output(n * n, 0.0f);

    const std::size_t m = 3;
    std::vector<float> mask(m * m);

    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = rand_num<float>(i_min, i_max);
    }

    for (std::size_t i = 0; i < m * m; ++i) {
        mask[i] = rand_num<float>(m_min, m_max);
    }

    omp_set_num_threads(t);

    double start = omp_get_wtime();
    convolve(image.data(), output.data(), n, mask.data(), m);
    double end = omp_get_wtime();
    double duration_msec = (end - start) * 1000.0;
    
    std::cout << output[0] << "\n";
    std::cout << output[n * n - 1] << "\n";
    std::cout << duration_msec.count() << "\n";

    return 0;
}