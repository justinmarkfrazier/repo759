#include "scan.h"
#include <cstddef>

// Performs an inclusive scan on input array arr and stores
// the result in the output array
// arr and output are arrays of n elements
void scan(const float *arr, float *output, std::size_t n)
{
    float sum_current{0.0f};
    for(std::size_t i = 0; i<n; ++i) {
        sum_current += arr[i];
        output[i] = sum_current;
    }
}