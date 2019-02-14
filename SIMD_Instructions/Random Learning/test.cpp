#include <immintrin.h>
#include <stdio.h>

int main() {

    __m256 reg;

    reg = 1.0;

    #if 0
    /* Compute the difference between the two vectors */
    __m256 result = _mm256_sub_ps(evens, odds);

    /* Display the elements of the result vector */
    float* f = (float*)&result;
    printf("%f %f %f %f %f %f %f %f\n",
    f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    #endif

    return 0;
}
