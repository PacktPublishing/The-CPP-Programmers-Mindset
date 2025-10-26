#include <gtest/gtest.h>

#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include "matmul.cuh"


struct RandomGenerator {
    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
        rng.discard(n); // advance RNG state by n
        return dist(rng);
    }
};

class MatrixMulTest : public ::testing::Test {
protected:
    cublasHandle_t handle = nullptr;

    void SetUp() override {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasCreate failed");
        };
    }

    void TearDown() override {
        if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasDestroy failed");
        }
    }



    thrust::device_vector<float> make_matrix(int m, int n) {
        unsigned size = m * n;
        thrust::device_vector<int> A(size);
        thrust::transform(
            thrust::counting_iterator<unsigned>(0),
            thrust::counting_iterator<unsigned>(size),
            A.begin(),
            RandomGenerator()
        );

        return A;
    }

    auto test_matmul(int m, int n, int k, float tolerance = 1e-4f) {

        const auto d_A = make_matrix(m, k);
        const auto d_B = make_matrix(k, n);
        thrust::device_vector<float> d_ct_C(m * n);
        thrust::device_vector<float> d_cb_C(m * n);

        // Run custom kernel
        constexpr int tile_size = 16;
        dim3 block(tile_size, tile_size);
        dim3 grid((n + tile_size - 1) / tile_size, (m + tile_size - 1) / tile_size);

        ct::matrix_mul<<<grid, block>>>(
            thrust::raw_pointer_cast(d_ct_C.data()),
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()),
            m, n, k
        );

        cudaError_t kernel_error = cudaGetLastError();
        ASSERT_EQ(kernel_error, cudaSuccess)
            << "Kernel launch failed: " << cudaGetErrorString(kernel_error);

        cudaError_t sync_error = cudaDeviceSynchronize();
        ASSERT_EQ(sync_error, cudaSuccess)
            << "Kernel execution failed: " << cudaGetErrorString(sync_error);

        // cuBLAS uses column-major, so we compute: C = B^T * A^T = (A * B)^T
        float alpha = 1.0f;
        float beta = 0.0f;
        ASSERT_EQ(cublasSgemm(
            handle,
            CUBLAS_OP_N,    // B is not transposed
            CUBLAS_OP_N,    // A is not transposed
            n,              // rows of result in column-major (cols in row-major)
            m,              // cols of result in column-major (rows in row-major)
            k,              // shared dimension
            &alpha,
            thrust::raw_pointer_cast(d_B.data()), n,  // B: k×n row-major = n×k col-major
            thrust::raw_pointer_cast(d_A.data()), k,  // A: m×k row-major = k×m col-major
            &beta,
            thrust::raw_pointer_cast(d_cb_C.data()), n  // C: m×n row-major = n×m col-major
        ), CUBLAS_STATUS_SUCCESS);

        cudaDeviceSynchronize();

        thrust::host_vector<float> h_C_custom = d_ct_C;
        thrust::host_vector<float> h_C_cublas = d_cb_C;

        for (int i = 0; i < m * n; ++i) {
            ASSERT_FLOAT_EQ(h_C_custom[i], h_C_cublas[i]);
        }

    }
};




TEST_F(MatrixMulTest, SmallSquare) {
    test_matmul(16, 16, 16);
}

TEST_F(MatrixMulTest, MediumSquare) {
    test_matmul(32, 32, 32);
}

TEST_F(MatrixMulTest, LargeSquare) {
    test_matmul(64, 64, 64);
}

TEST_F(MatrixMulTest, VeryLargeSquare) {
    test_matmul(128, 128, 128);
}

TEST_F(MatrixMulTest, HugeSquare) {
    test_matmul(256, 256, 256);
}

TEST_F(MatrixMulTest, RectangularWide) {
    test_matmul(64, 128, 32);
}

TEST_F(MatrixMulTest, RectangularTall) {
    test_matmul(128, 64, 96);
}

TEST_F(MatrixMulTest, RectangularVeryWide) {
    test_matmul(32, 256, 64);
}

TEST_F(MatrixMulTest, RectangularVeryTall) {
    test_matmul(256, 32, 64);
}

TEST_F(MatrixMulTest, LargeK) {
    test_matmul(64, 64, 256);
}
