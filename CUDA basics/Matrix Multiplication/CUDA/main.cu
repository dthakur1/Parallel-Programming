#include <iostream>
#include <cstddef>
#include <cstdio>
#include <random>
#include <chrono>
#include <cassert>

using std::size_t;

constexpr size_t N = 2048;
constexpr size_t BLOCK_SIZE = 16;
using array_t = float [N][N];

bool
is_equal(const array_t &a1, const array_t &a2) {
    
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (std::abs(a1[i][j] - a2[i][j]) > 0.0001) {
                fprintf(stderr, "At (%zu, %zu): %f != %f\n",
                 i, j, a1[i][j], a2[i][j]);
                return false;
            }
        }
    }

    return true;
}

void
zero(array_t &a) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            a[i][j] = 0;
        }
    }
}

void
blocked(const array_t &a, const array_t &b, array_t &c) {

    // Blocked.
    for (size_t b_i = 0;  b_i < N/BLOCK_SIZE; b_i++) {
        for (size_t b_j = 0;  b_j < N/BLOCK_SIZE; b_j++) {

            // Zero out that block.
            for (size_t i = BLOCK_SIZE*b_i; i < BLOCK_SIZE*(b_i + 1); i++) {
                for (size_t j = BLOCK_SIZE*b_j; j < BLOCK_SIZE*(b_j + 1); j++) {
                    c[i][j] = 0;
                }
            }

            for (size_t b_k = 0; b_k < N/BLOCK_SIZE; b_k++) {

                for (size_t i = BLOCK_SIZE*b_i; i < BLOCK_SIZE*(b_i + 1); i++) {
                    for (size_t j = BLOCK_SIZE*b_j; j < BLOCK_SIZE*(b_j + 1); j++) {
                        for (size_t k = 0; k < BLOCK_SIZE; k++) {
                            /*
                            fprintf(stderr, "c[%zu][%zu] += a[%zu][%zu]*b[%zu][%zu]\n",
                             i, j,
                             i, BLOCK_SIZE*b_k + k,
                             BLOCK_SIZE*b_k + k,
                             j);
                            */
                            c[i][j] += a[i][BLOCK_SIZE*b_k + k]*b[BLOCK_SIZE*b_k + k][j];
                        }
                    }
                }
            }
        }
    }
}

void
nonblocked(const array_t &a, const array_t &b, array_t &c) {

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            c[i][j] = 0;
            for (size_t k = 0; k < N; k++) {
                c[i][j] += a[i][k]*b[k][j];
            }
        }
    }
}

__global__ void
nonblocked_gpu(const float *a, const float *b, float *c) {

    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    float x = 0;
    for (int k = 0; k < N; k++) {
        // x += a[i][k]*b[k][j];
        x += a[N*i + k]*b[N*k + j];
    }
    c[N*i + j] = x;
}

__global__ void
blocked_gpu(const float *a, const float *b, float *c) {

    __shared__ float a_block[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_block[BLOCK_SIZE][BLOCK_SIZE];

    // These are block indices relative to the grid.
    int b_i = blockIdx.y, b_j = blockIdx.x;
    // These indices are relative to the block.
    int t_i = threadIdx.y, t_j = threadIdx.x;

    // These are relative to whole matrix.
    int i = b_i*BLOCK_SIZE + t_i;
    int j = b_j*BLOCK_SIZE + t_j;

    float x = 0;

    for (int h = 0; h < N/BLOCK_SIZE; h++) {

        /*
         * Each thread loads one element in each of a and b blocks.
         */

        // Load a[i][BLOCK_SIZE*h + t_j]
        a_block[t_i][t_j] = a[N*i + BLOCK_SIZE*h + t_j];
        // Load b[BLOCK_SIZE*h + t_i][j];
        b_block[t_i][t_j] = b[N*(BLOCK_SIZE*h + t_i) + j];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            x += a_block[t_i][k]*b_block[k][t_j];
        }

        __syncthreads();
    }

    c[N*i + j] = x;
}

int
main() {

    cudaError_t rv_ce;

    assert(N%BLOCK_SIZE == 0);

    printf("Total memory for arrays: %zu\n", 8*3*N*N);
    printf("Total memory for blocks: %zu\n", 8*3*BLOCK_SIZE*BLOCK_SIZE);

    std::minstd_rand eng;
    std::uniform_real_distribution<float> dist(-1, 1);

    //static array_t a, b, c, check;
    static array_t a, b, c, check;

    fprintf(stderr, "Creating a, b arrays...\n");

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            a[i][j] = dist(eng);
            b[i][j] = dist(eng);
        }
    }

    fprintf(stderr, "Creating check array...\n");
    // Compute result using the usual way for verification of correctness.
    // nonblocked(a, b, check);

    /*
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            fprintf(stderr, "a[%zu][%zu] == %f, b[%zu][%zu] == %f\n",
             i, j, a[i][j],
             i, j, b[i][j]);
        }
    }
    */

    float *d_a, *d_b, *d_c;

    fprintf(stderr, "Resetting...\n");
    rv_ce = cudaDeviceReset();
    assert(rv_ce == cudaSuccess);

    fprintf(stderr, "Allocating...\n");

    // Allocate memory.
    rv_ce = cudaMalloc((void **) &d_a, sizeof(float)*N*N);
    assert(rv_ce == cudaSuccess);
    rv_ce = cudaMalloc((void **) &d_b, sizeof(float)*N*N);
    assert(rv_ce == cudaSuccess);
    rv_ce = cudaMalloc((void **) &d_c, sizeof(float)*N*N);
    assert(rv_ce == cudaSuccess);

    fprintf(stderr, "Copying...\n");

    // Copy arrays to GPU.
    rv_ce = cudaMemcpy(d_a, (void *) a, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    assert(rv_ce == cudaSuccess);
    rv_ce = cudaMemcpy(d_b, (void *) b, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    assert(rv_ce == cudaSuccess);

    /*
     * Test performance of nonblocked.
     */

    {
        fprintf(stderr, "Kernel start...\n");

        // Call the kernel.
        auto start = std::chrono::high_resolution_clock::now();

        dim3 grid_dim{N/BLOCK_SIZE, N/BLOCK_SIZE, 1}, block_dim{BLOCK_SIZE, BLOCK_SIZE, 1};
        nonblocked_gpu<<<grid_dim, block_dim>>>(d_a, d_b, d_c);
        assert(cudaPeekAtLastError() == cudaSuccess);
        assert(cudaDeviceSynchronize() == cudaSuccess);

        std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - start;
        std::cout << (2*N*N*N)/dt.count() << " FLOPS" << std::endl;

        printf("Kernel stop.\n");

        rv_ce = cudaMemcpy((void *) c, d_c, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
        assert(rv_ce == cudaSuccess);

        /*
        if (!is_equal(*(array_t *) c, check)) {
            fprintf(stderr, "Check FAILED!\n");
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    // fprintf(stderr, "c[%zu][%zu] is %f\n", i, j, c[i][j]);
                }
            }
        }
        */
    }

    /*
     * Test performance of blocked.
     */

    {
        fprintf(stderr, "Kernel start...\n");

        // Erase results.
        rv_ce = cudaMemset(d_c, 0, sizeof(float)*N*N);
        assert(rv_ce == cudaSuccess);

        // Call the kernel.
        auto start = std::chrono::high_resolution_clock::now();

        dim3 grid_dim{N/BLOCK_SIZE, N/BLOCK_SIZE, 1}, block_dim{BLOCK_SIZE, BLOCK_SIZE, 1};
        blocked_gpu<<<grid_dim, block_dim>>>(d_a, d_b, d_c);
        assert(cudaPeekAtLastError() == cudaSuccess);
        assert(cudaDeviceSynchronize() == cudaSuccess);

        std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - start;
        std::cout << (2*N*N*N)/dt.count() << " FLOPS" << std::endl;

        printf("Kernel stop.\n");

        rv_ce = cudaMemcpy((void *) c, d_c, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
        assert(rv_ce == cudaSuccess);

        /*
        if (!is_equal(*(array_t *) c, check)) {
            fprintf(stderr, "Check FAILED!\n");
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    // fprintf(stderr, "c[%zu][%zu] is %f\n", i, j, c[i][j]);
                }
            }
        }
        */
    }

    fprintf(stderr, "Free...\n");

    rv_ce = cudaFree(d_a);
    assert(rv_ce == cudaSuccess);
    rv_ce = cudaFree(d_b);
    assert(rv_ce == cudaSuccess);
    rv_ce = cudaFree(d_c);
    assert(rv_ce == cudaSuccess);


    #if 0
    /*
     * Nonblocked matrix multiply.
     */

    // Warmup.
    nonblocked(a, b, c);

    {
        auto start = std::chrono::high_resolution_clock::now();
        nonblocked(a, b, c);
        std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - start;
        std::cout << (2*N*N*N)/dt.count() << " FLOPS" << std::endl;
        assert(is_equal(c, check));
    }

    /*
     * Blocked matrix multiply.
     */

    zero(c);

    // Warmup.
    blocked(a, b, c);

    {
        auto start = std::chrono::high_resolution_clock::now();
        blocked(a, b, c);
        std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - start;
        std::cout << (2*N*N*N)/dt.count() << " FLOPS" << std::endl;
        assert(is_equal(c, check));
    }

    #endif
}
