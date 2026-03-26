#include "montecarlo.h"

int montecarlo(const size_t n, const float *x, const float *y, const float radius) {
    int incircle = 0;
    const float r2 = radius * radius;

    #pragma omp parallel for simd reduction(+:incircle)
    for (size_t i = 0; i < n; ++i) {
        const float xi = x[i];
        const float yi = y[i];
        incircle += (xi * xi + yi * yi <= r2);
    }

    return incircle;
}