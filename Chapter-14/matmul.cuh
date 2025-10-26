#ifndef CHAPTER_14_MATMUL_CUH
#define CHAPTER_14_MATMUL_CUH


namespace ct {
__global__ void matrix_mul(
    float *__restrict__ C,
    const float *__restrict__ A,
    const float *__restrict__ B,
    int m, int n, int k);


}

#endif //CHAPTER_14_MATMUL_CUH