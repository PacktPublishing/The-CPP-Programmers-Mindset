

#include "saxpy_mt.hpp"


[[gnu::target_clones("avx2,default")]]
void ct::cpu_saxpy(float a, const float* x, float* y, int N) {
    for (int i=0; i<N; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

[[gnu::target_clones("avx2,default")]]
void ct::cpu_mt_saxpy(float a, const float *x, float *y, int N) {
#pragma omp parallel for num_threads(4) default(none) shared(a, x, y, N)
    for (int i=0; i<N; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

void ct::openmp_saxpy(float a, const float* x, float* y, int N) {
#pragma omp target teams distribute parallel for \
    default(none) shared(a, N, x, y)\
    map(to:x[0:N]) \
    map(tofrom:y[0:N])
    for (int i=0; i<N; ++i) {
        y[i] = a*x[i] + y[i];
    }

}
