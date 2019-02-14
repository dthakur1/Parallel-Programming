#include <iostream>
#include <cstddef>
#include <cstdio>
#include <random>
#include <chrono>
#include <cassert>

using std::size_t;

constexpr size_t N = 1024;
constexpr size_t BLOCK_SIZE = 32;
static_assert(N%BLOCK_SIZE == 0);
using array_t = double [N][N];

bool
is_equal(const array_t &a1, const array_t &a2) {
    
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (a1[i][j] != a2[i][j]) {
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

int
main() {

    printf("Total memory for arrays: %zu\n", 8*3*N*N);
    printf("Total memory for blocks: %zu\n", 8*3*BLOCK_SIZE*BLOCK_SIZE);

    std::minstd_rand eng;
    std::uniform_real_distribution<double> dist(-1, 1);

    static array_t a, b, c, check;

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            a[i][j] = dist(eng);
            b[i][j] = dist(eng);
        }
    }

    // Compute result using the usual way for verification of correctness.
    nonblocked(a, b, check);

    /*
     * Nonblocked matrix multiply.
     */

    // Warmup.
    nonblocked(a, b, c);

    {
        auto start = std::chrono::high_resolution_clock::now();
        nonblocked(a, b, c);
        std::chrono::duration<double> dt = std::chrono::high_resolution_clock::now() - start;
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
        std::chrono::duration<double> dt = std::chrono::high_resolution_clock::now() - start;
        std::cout << (2*N*N*N)/dt.count() << " FLOPS" << std::endl;
        assert(is_equal(c, check));
    }
}
