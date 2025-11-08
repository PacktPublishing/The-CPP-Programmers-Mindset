#ifndef CHAPTER_14_SAXPY_CUH
#define CHAPTER_14_SAXPY_CUH
#include <thrust/device_vector.h>


namespace ct {


void thrust_saxpy(float a,
    const thrust::device_vector<float>& x,
    const thrust::device_vector<float>& y,
    thrust::device_vector<float>& out
    );

__global__ void saxpy(float a, const float* __restrict__ x, float* __restrict__ y, int N);

template <typename Scalar>
class AxpyKernel {
public:
    struct Params {
        const Scalar* x;
        Scalar* y;
        Scalar a;
        int N;
    };

    __device__ __forceinline__
    void operator()(const Params& params) {
        int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;

        for (int i=thread_rank; i<params.N; i+= stride) {
            params.y[i] = params.a*params.x[i] + params.y[i];
        }
    }
};

template <typename Kernel>
__global__ void kernel(typename Kernel::Params params) {
    Kernel kernel;

    kernel(params);
}


// using the saxpy kernel
cudaError_t basic_saxpy(float a, const float* d_x, float* d_y, int N);

// using the object defined above
cudaError_t kernel_object_saxpy(float a, const float* d_x, float* d_y, int N);


}


#endif //CHAPTER_14_SAXPY_CUH