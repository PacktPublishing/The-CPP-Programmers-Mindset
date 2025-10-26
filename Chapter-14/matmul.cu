#include "matmul.cuh"

__global__ void ct::matrix_mul(
    float * __restrict__ C, // m by n matrix, row major
    const float * __restrict__ A, // m by k matrix, row major
    const float * __restrict__ B, // k by n matrix, row major
    int m, int n, int k) {

    constexpr int tile_size = 16; // block is tile_size by tile_size
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float A_tile[tile_size][tile_size];
    __shared__ float B_tile[tile_size][tile_size];

    // Tiles move across rows of A
    const float* sub_A = A + (blockIdx.y * tile_size * k);
    // Tiles move down columns of B
    const float* sub_B = B + (blockIdx.x * tile_size);

    float* sub_C = C + (blockIdx.y*tile_size*n) + blockIdx.x * tile_size;

    float accumulator = 0.0f;
    for (int tile_k=0; tile_k < gridDim.x; ++tile_k) {
        // read into shared memory tiles
        // move A_tile over by tile_size rows
        auto A_offset = tile_k * tile_size;
        // move B_tile down by tile_size cols
        auto B_offset = tile_k * tile_size * n;

        A_tile[ty][tx] = sub_A[A_offset + ty*k + tx];
        B_tile[ty][tx] = sub_B[B_offset + ty*n + tx];

        // Do not proceed until the memory is copied
        __syncthreads();

        for (int i=0; i<tile_size; ++i) {
            accumulator += A_tile[ty][i] * B_tile[i][tx];
        }
        __syncthreads();

    }

    // write out the result
    sub_C[ty*n + tx] = accumulator;
}
