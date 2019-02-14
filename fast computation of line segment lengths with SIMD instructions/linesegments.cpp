#include <immintrin.h>
#include <cmath>
#include <functional>
#include <chrono>
#include <random>
#include <iostream>
#include <cassert>

const int N = 16*1000000;

double
time(const std::function<void ()> &f) {
    f(); // Run once to warmup.
    // Now time it for real.
    auto start = std::chrono::system_clock::now();
    f();
    auto stop = std::chrono::system_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

int
main() {

    alignas(32) static float x[N], y[N], z[N], a[N];
    alignas(32) static float x2[N], y2[N], z2[N], a2[N];

    /*
     * Generate data.
     */

    std::default_random_engine eng;
    std::uniform_real_distribution<float> dist(-1, 1);
    for (int i = 0; i < N; i++) {
        x[i] = dist(eng);
        y[i] = dist(eng);
        z[i] = dist(eng);
        a[i] = dist(eng);

        x2[i] = dist(eng);
        y2[i] = dist(eng);
        z2[i] = dist(eng);
        a2[i] = dist(eng);
    }

    /*
     * Sequential.
     */

    static float l_s[N];
    auto seq = [&]() {
        for (int i = 0; i < N; i++) {
            l_s[i] = std::sqrt((x[i]-x2[i]) *(x[i]-x2[i]) + (y[i]-y2[i]) *(y[i]-y2[i]) +
                                (z[i]-z2[i]) *(z[i]-z2[i]) + (a[i]-a2[i]) *(a[i]-a2[i]) ) ;
        }
    };

    std::cout << "Sequential: " << (N/time(seq))/1000000 << " Mops/s" << std::endl;

    alignas(32) static float l_v[N];
    auto vec = [&]() {
        for (int i = 0; i < N/8; i++) {
            __m256 ymm_x = _mm256_load_ps(x + 8*i);
            __m256 ymm_x2= _mm256_load_ps(x2 + 8*i);
            __m256 ymm_difx = _mm256_sub_ps ( ymm_x, ymm_x2);

            __m256 ymm_y = _mm256_load_ps(y + 8*i);
            __m256 ymm_y2= _mm256_load_ps(y2 + 8*i);
            __m256 ymm_dify = _mm256_sub_ps ( ymm_y, ymm_y2);

            __m256 ymm_z = _mm256_load_ps(z + 8*i);
            __m256 ymm_z2= _mm256_load_ps(z2 + 8*i);
            __m256 ymm_difz = _mm256_sub_ps ( ymm_z, ymm_z2);

            __m256 ymm_a = _mm256_load_ps(a + 8*i);
            __m256 ymm_a2= _mm256_load_ps(a2 + 8*i);
            __m256 ymm_difa = _mm256_sub_ps ( ymm_a, ymm_a2);

            __m256 ymm_l = _mm256_sqrt_ps
                            (
                            _mm256_mul_ps(ymm_difx, ymm_difx) 
                            + _mm256_mul_ps(ymm_dify, ymm_dify) 
                            + _mm256_mul_ps(ymm_difz, ymm_difz)
                            + _mm256_mul_ps(ymm_difa, ymm_difa) 
                            );

            _mm256_store_ps(l_v + 8*i, ymm_l);
        }
    };

    std::cout << "Vector: " << (N/time(vec))/1000000 << " Mops/s" << std::endl;

    for (int i = 0; i < N; i++) {
        if (l_s[i] - l_v[i] != 0) {
            assert(false);
        }
    }
}