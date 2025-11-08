
#include <cstdio>
#include <iostream>

#include <thrust/universal_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/execution_policy.h>

#include "cooperative_groups.h"

namespace cg = cooperative_groups;

__global__ void compute_2norm(const float *vectors,
                              float *norms,
                              unsigned n_vectors) {
    constexpr unsigned vector_size = 32;

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    // For simplicity, assume that a block is 32 threads
    const unsigned my_vector_offset = grid.block_rank() * vector_size;

    float my_value = vectors[my_vector_offset + warp.thread_rank()];
    float my_norm = my_value * my_value;

    for (int offset = 16; offset > 0; offset >>= 1) {
        my_norm += warp.shfl_down(my_norm, offset);
    }

    // only thread 0 writes to memory
    if (threadIdx.x == 0) {
        norms[blockIdx.x] = sqrtf(my_norm);
    }
}


int main() {
    auto num_vectors = 256;
    thrust::universal_vector<float> vectors(num_vectors * 32);
    thrust::universal_vector<float> norms(num_vectors);

    thrust::random::default_random_engine rng;
    thrust::random::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    thrust::generate(thrust::host, vectors.begin(), vectors.end(), [&] { return dist(rng); });

    compute_2norm<<<num_vectors, 32>>>(
        raw_pointer_cast(vectors.data()),
        raw_pointer_cast(norms.data()),
        num_vectors
    );
    cudaDeviceSynchronize();

    std::cout << "Norms (" << norms.size() << "): \n";
    for (const auto &norm: norms) {
        std::cout << norm << "\n";
    }
    std::cout << "Done\n";
}
