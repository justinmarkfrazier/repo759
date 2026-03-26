#include "cluster.h"
#include <cmath>

void cluster(const size_t n, const size_t t, const float *arr,
             const float *centers, float *dists) {
#pragma omp parallel num_threads(t)
  {
    int tid = omp_get_thread_num();
    float local_dist = 0.0f;
    float center = centers[tid];

#pragma omp for schedule(static)
    for (size_t i = 0; i < n; i++) {
      local_dist += std::fabs(arr[i] - center);
    }

    dists[tid] = local_dist;
  }
}