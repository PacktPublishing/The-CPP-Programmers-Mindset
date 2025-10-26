#include "saxpy.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>


void ct::thrust_saxpy(float a, const thrust::device_vector<float> &x, const thrust::device_vector<float> &y,
    thrust::device_vector<float> &out) {

    thrust::transform(
        x.begin(), x.end(),
        y.begin(),
        out.begin(),
        [a] __device__ (float xe, float ye) {
            return a * xe + ye;
        });

}


__global__ void ct::saxpy(float a, const float* __restrict__ x, float* __restrict__ y, int N) {
    int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = thread_rank; i < N; i += stride) {
        y[i] = a * x[i] + a * y[i];
    }
}

cudaError_t ct::basic_saxpy(float a, const float *d_x, float *d_y, int N) {

    dim3 block { 256 };
    dim3 grid {
        ( N + block.x - 1 ) / block.x // ceil div
    };

    saxpy<<<grid, block>>>(a, d_x, d_y, N);

    return cudaGetLastError();
}

cudaError_t ct::kernel_object_saxpy(float a, const float *d_x, float *d_y, int N) {
    using Kernel = AxpyKernel<float>;

    dim3 block { 256 };
    dim3 grid {
        ( N + block.x - 1 ) / block.x // ceil div
    };

    typename Kernel::Params params {
        d_x,
        d_y,
        a,
        N
    };

    kernel<Kernel><<<grid, block>>>(params);

    return cudaGetLastError();
}
