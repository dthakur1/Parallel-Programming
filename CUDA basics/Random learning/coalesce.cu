#include <iostream>
#include <cstddef>
#include <cstdio>
#include <random>
#include <chrono>
#include <cassert>
#include <vector>

using std::size_t;

#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
void
gpu_assert_h(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ unsigned long long dev_sum;

const int N = 512*1024*1024;

__global__ void
sum_noncoalesced(const int *const array) {

    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int chunk_size = N/(gridDim.x*blockDim.x);
    // printf("chunk_size %u\n", chunk_size);

    const int *const chunk_start = array + chunk_size*tid;
    const int *const chunk_end = array + chunk_size*(tid + 1);
    for (const int *p = chunk_start; p < chunk_end; p++) {
        atomicAdd(&dev_sum, *p);
    }
}

__global__ void
sum_noncoalesced_shared(const int *const array) {

    __shared__ unsigned long long shared;
    if (threadIdx.x == 0) {
        shared = 0;
    }
    __syncthreads();

    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int chunk_size = N/(gridDim.x*blockDim.x);
    // printf("chunk_size %u\n", chunk_size);

    const int *const chunk_start = array + chunk_size*tid;
    const int *const chunk_end = array + chunk_size*(tid + 1);
    for (const int *p = chunk_start; p < chunk_end; p++) {
        atomicAdd(&shared, *p);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&dev_sum, shared);
    }
}

__global__ void
sum_noncoalesced_local(const int *const array) {

    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int chunk_size = N/(gridDim.x*blockDim.x);
    // printf("chunk_size %u\n", chunk_size);

    const int *const chunk_start = array + chunk_size*tid;
    const int *const chunk_end = array + chunk_size*(tid + 1);
    unsigned long long sum = 0;
    for (const int *p = chunk_start; p < chunk_end; p++) {
        sum += *p;
    }

    atomicAdd(&dev_sum, sum);
}

__global__ void
sum_noncoalesced_local_shared(const int *const array) {

    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ unsigned long long shared_sum;

    if (threadIdx.x == 0) {
        shared_sum = 0;
    }
    __syncthreads();

    unsigned int chunk_size = N/(gridDim.x*blockDim.x);
    // printf("chunk_size %u\n", chunk_size);

    const int *const chunk_start = array + chunk_size*tid;
    const int *const chunk_end = array + chunk_size*(tid + 1);
    unsigned long long sum = 0;
    for (const int *p = chunk_start; p < chunk_end; p++) {
        sum += *p;
    }

    atomicAdd(&shared_sum, sum);

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&dev_sum, shared_sum);
    }
}

__global__ void
sum_coalesced_local_shared(const int *const array) {

    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ unsigned long long shared_sum;

    if (threadIdx.x == 0) {
        // printf("Thread %u: before start dev sum: %llu\n", tid, dev_sum);
        shared_sum = 0;
    }
    __syncthreads();

    // printf("Thread: %u\n", tid);

    // Stride is the number of threads.
    const unsigned int stride = gridDim.x*blockDim.x;
    if (tid == 0) {
        printf("Stride %u\n", stride);
    }

    const int *const start = array + tid;
    const int *const end = array + N;
    unsigned long long sum = 0;
    for (const int *p = start; p < end; p += stride) {
        sum += *p;
        // printf("Thread %02u added element %02d\n", tid, (int) (p - array));
    }
    // printf("Thread %u: local sum: %llu\n", tid, sum);

    atomicAdd(&shared_sum, sum);

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&dev_sum, shared_sum);
    }
}

void
time(const char *name, void (*kernel)(const int *const), const int *const d_array, const dim3 &grid, const dim3 &block) {

    cudaError_t rv_ce;

    // Clearing sum.
    const unsigned long long zero = 0;
    rv_ce = cudaMemcpyToSymbol(dev_sum, &zero, sizeof(unsigned long long));
    gpu_assert(rv_ce);

    /*
    fprintf(stderr, "%s kernel start {%u, %u, %u}, {%u, %u, %u}...\n", name,
     grid.x, grid.y, grid.z, block.x, block.y, block.z);
    */

    // Call the kernel.
    auto start = std::chrono::high_resolution_clock::now();

    // dim3 grid_dim{16, 1, 1}, block_dim{256, 1, 1};
    (*kernel)<<<grid, block>>>(d_array);
    assert(cudaPeekAtLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - start;
    // std::cout << N/dt.count() << " op/s" << ", dt = " << dt.count() << std::endl;

    fprintf(stderr, "%s with {%u, %u, %u}, {%u, %u, %u}: %e op/s, %f time\n", name,
     grid.x, grid.y, grid.z, block.x, block.y, block.z,
     N/dt.count(),
     dt.count()
    );

    // printf("Kernel stop.\n");

    // Copy answer from GPU to host.
    unsigned long long sum_check;
    rv_ce = cudaMemcpyFromSymbol(&sum_check, dev_sum, sizeof(unsigned long long));
    gpu_assert(rv_ce);
    
    // printf("check: %lld\n", sum_check);
}

int
main() {

    cudaError_t rv_ce;

    std::minstd_rand eng;
    std::uniform_int_distribution<int> dist(0, 1000000);

    std::vector<int> array(N);

    fprintf(stderr, "Creating data...\n");

    for (size_t i = 0; i < N; i++) {
        int k = dist(eng);
        array.at(i) = k;
    }

    unsigned long long sum = 0;

    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < N; i++) {
            sum += array[i];
        }
        std::chrono::duration<float> dt = std::chrono::high_resolution_clock::now() - start;
        std::cout << "CPU: " << N/dt.count() << " op/s" << ", dt = " << dt.count() << std::endl;
    }
    printf("sum: %llu\n", sum);

    fprintf(stderr, "Resetting...\n");
    rv_ce = cudaDeviceReset();
    assert(rv_ce == cudaSuccess);

    fprintf(stderr, "Allocating...\n");

    // Allocate memory.
    int *d_array;
    rv_ce = cudaMalloc((void **) &d_array, sizeof(int)*N); gpu_assert(rv_ce);

    fprintf(stderr, "Copying...\n");

    // Copy array to GPU.
    rv_ce = cudaMemcpy(d_array, (void *) array.data(), sizeof(int)*N, cudaMemcpyHostToDevice);
    assert(rv_ce == cudaSuccess);

    // time("Noncoalesced, atomic global",  &sum_noncoalesced, d_array, {16, 1, 1}, {256, 1, 1});
    /*
    time("Noncoalesced, atomic shared", &sum_noncoalesced_shared, d_array);
    time("Noncoalesced, local", &sum_noncoalesced_local, d_array);
    time("Coalesced, local", &sum_coalesced_local, d_array);
    */

    for (unsigned int grid = 16; grid <= 64; grid *= 2) {
        for (unsigned int block = 64; block <= 1024; block *= 2) {
            time("Coalesced, local-shared",  &sum_coalesced_local_shared, d_array, {grid, 1, 1}, {block, 1, 1});
        }
    }

    for (unsigned int grid = 32; grid <= 64; grid *= 2) {
        for (unsigned int block = 256; block <= 1024; block *= 2) {
            time("Noncoalesced, local-shared",  &sum_noncoalesced_local_shared, d_array, {grid, 1, 1}, {block, 1, 1});
        }
    }

    #if 0
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

    #endif


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
