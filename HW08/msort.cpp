#include "msort.h"
#include <algorithm>
#include <vector>

static void merge_arrays(int* arr, std::size_t n) {
    std::size_t mid = n / 2;
    std::size_t i = 0;
    std::size_t j = mid;
    std::size_t k = 0;

    std::vector<int> temp(n);

    while (i < mid && j < n) {
        if (arr[i] <= arr[j]) {
            temp[k] = arr[i];
            ++i;
        } else {
            temp[k] = arr[j];
            ++j;
        }
        ++k;
    }

    while (i < mid) {
        temp[k] = arr[i];
        ++i;
        ++k;
    }

    while (j < n) {
        temp[k] = arr[j];
        ++j;
        ++k;
    }

    for (std::size_t p = 0; p < n; ++p) {
        arr[p] = temp[p];
    }
}

static void msort_recursive(int* arr, std::size_t n, std::size_t threshold) {
    if (n <= 1) {
        return;
    }

    if (n <= threshold) {
        std::sort(arr, arr + n);
        return;
    }

    std::size_t mid = n / 2;
    int* right = arr + mid;
    std::size_t right_size = n - mid;

    #pragma omp task
    msort_recursive(arr, mid, threshold);

    #pragma omp task
    msort_recursive(right, right_size, threshold);

    #pragma omp taskwait

    merge_arrays(arr, n);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            msort_recursive(arr, n, threshold);
        }
    }
}