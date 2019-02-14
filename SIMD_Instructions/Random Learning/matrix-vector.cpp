#include <immintrin.h>
#include <cmath>
#include <functional>
#include <chrono>
#include <random>
#include <iostream>
#include <cassert>
#include <iomanip>

// Do N matrix-vector multiplications.
const int N = 1'000'000;

double
time(const std::function<void ()> &f) {
    f(); // Run once to warmup.
    // Now time it for real.
    auto start = std::chrono::system_clock::now();
    f();
    auto stop = std::chrono::system_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

void
print_vector(const float (&v)[8], int indent) {
    std::cout << std::string(indent, ' ');
    for (int i = 0; i < 8; i++) {
        std::cout << std::fixed << std::setprecision(6) << std::setw(9) << std::setfill(' ') << v[i];
        if (i < 7) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
}

void
print_matrix(const float (&m)[8][8], int indent) {
    for (int i = 0; i < 8; i++) {
        std::cout << std::string(indent, ' ') << "row " << i << ": ";
        print_vector(m[i], 0);
    }
}

int
main() {

    /*
     * Generate data.
     */

    std::default_random_engine eng;
    std::uniform_real_distribution<float> dist(-1, 1);

    // Generate vectors.

    // These are already "transposed".  In other words, they should be column
    // vectors, but each vector is actually a row vector.
    alignas(32) static float vecs[N][8];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 8; j++) {
            vecs[i][j] = dist(eng);
            // vecs[i][j] = j;
        }
    }
    /*
    std::cout << "vectors: " << std::endl;
    for (int i = 0; i < N; i++) {
        print_vector(vecs[i], 4);
    }
    */

    // Generate the matrix.
    alignas(32) float m[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            m[i][j] = dist(eng);
            // m[i][j] = 8*i + j;
        }
    }
    std::cout << "matrix:" << std::endl;
    print_matrix(m, 4);

    /*
     * Sequential test.
     */

    static float results_seq[N][8];
    auto sequential_test = [&]() {
        // Iterate over all vectors.
        for (int v_ind = 0; v_ind < N; v_ind++) {
            // Do the matrix-vector multiply.
            for (int i = 0; i < 8; i++) {
                results_seq[v_ind][i] = 0;
                for (int j = 0; j < 8; j++) {
                    results_seq[v_ind][i] += m[i][j]*vecs[v_ind][j];
                }
            }
        }
    };

    std::cout << "Sequential: " << (N/time(sequential_test))/1000000 << " Mops/s" << std::endl;

    /*
    std::cout << "Sequential results: " << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << "    " << i << ": ";
        print_vector(results_seq[i], 4);
    }
    */

    /*
     * Vector test.
     */

    alignas(32) static float results_vec[N][8];

    // Tranpose the matrix.
    alignas(32) float m_T[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            m_T[i][j] = m[j][i];
        }
    }
    /*
    // Print out the transposed matrix.
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            std::cout << m_T[i][j];
            if (j < 7) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
    */

    auto vector_test = [&]() {
        // Load rows of the transposed matrix into ymm registers representing the columns of the original.
        __m256 ymm_cols[8];
        for (int i = 0; i < 8; i++) {
            ymm_cols[i] = _mm256_load_ps(m_T[i]);
        }

        // Iterate over the vectors.
        for (int i = 0; i < N; i++) {
            __m256 ymm_result = _mm256_setzero_ps();
            for (int j = 0; j < 8; j++) {
                // Set all of ymm_v to the corresponding vector element.
                __m256 ymm_v = _mm256_broadcast_ss(&vecs[i][j]);
                // Multiply-add.
                ymm_result = _mm256_add_ps(ymm_result, _mm256_mul_ps(ymm_cols[j], ymm_v));
            }
            _mm256_store_ps(results_vec[i], ymm_result);
        }
    };

    std::cout << "vector test: " << (N/time(vector_test))/1000000 << " Mops/s" << std::endl;

    /*
    std::cout << "vector results:" << std::endl;
    for (int i = 0; i < N; i++) {
        print_vector(results_vec[i], 4);
    }
    */

    /*
     * Verify.
     */

    // float diff[8];
    // std::cout << "difference: " << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 8; j++) {
            // diff[j] = results_seq[i][j] - results_vec[i][j];
            assert(results_seq[i][j] - results_vec[i][j] == 0);
        }
        // print_vector(diff, 4);
    }
}
