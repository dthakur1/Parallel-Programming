#include <omp.h>
#include <iostream>
#include <stdio.h>
#include <random>
#include <array>

int
main() {

    std::uniform_real_distribution<float> dist(0, 1);

    std::array<float, 100> check;
    std::vector<float> big(5'000'000'000);

    #pragma omp parallel
    {
        std::default_random_engine eng(omp_get_thread_num());
        printf("Thread %d: %f\n", omp_get_thread_num(), dist(eng));
        #pragma omp for
            for (std::size_t i = 0; i < check.size(); i++) {
                check[i] = dist(eng);
            }
        #pragma omp for
            for (std::size_t i = 0; i < big.size(); i++) {
                big[i] = dist(eng);
            }
    }

    // Print out for manual inspection.
    for (auto x : check) {
        std::cout << x << std::endl;
    }

    // Verify computationally.
    double avg = 0;
    #pragma omp parallel for reduction(+:avg)
        for (std::size_t i = 0; i < big.size(); i++) {
            avg += big[i];
        }
    avg /= big.size();
    std::cout << "average: " << avg << std::endl;
}