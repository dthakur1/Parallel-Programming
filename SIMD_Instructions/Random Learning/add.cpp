#include <immintrin.h>
#include <iostream>
#include <cassert>

int
main() {

    alignas(32) static float
     x[8] = {1, 2, 3, 4, 5, 6, 7, 8},
     y[8] = {.1, .2, .3, .4, .5, .6, .7, .8},
     z[8];

    __m256 ymm_x = _mm256_load_ps(x);
    __m256 ymm_y = _mm256_load_ps(y);
    __m256 ymm_z = _mm256_add_ps(ymm_x, ymm_y);
    _mm256_store_ps(z, ymm_z);

    for (auto v : z) {
        std::cout << v << std::endl;
    }
}
