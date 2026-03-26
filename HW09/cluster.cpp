#include "cluster.h"
#include <cmath>
#include <omp.h>

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
    const size_t PAD = 16;

#pragma omp parallel num_threads(t)
    {
        size_t tid = omp_get_thread_num();

#pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++) {
            dists[tid * PAD] += std::fabs(arr[i] - centers[tid * PAD]);
        }
    }
}