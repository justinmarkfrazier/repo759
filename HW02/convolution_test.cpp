#include "convolution.h"
#include <iostream>
#include <cstddef>


int main()
{
    const std::size_t n = 4;
    const std::size_t m = 3;

    float image[n * n] = {
        1, 3, 4, 8,
        6, 5, 2, 4,
        3, 4, 6, 8,
        1, 4, 5, 2
    };

    float mask[m * m] = {
        0, 0, 1,
        0, 1, 0,
        1, 0, 0
    };

    float output[n * n] = {0};

    convolve(image, output, n, mask, m);

    // Print the output
    for(std::size_t i = 0; i < n; ++i) {
        for(std::size_t j = 0; j < n; ++j) {
            std::cout << output[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}