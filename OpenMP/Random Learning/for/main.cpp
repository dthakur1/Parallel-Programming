#include <omp.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <functional>
#include <chrono>
#include <random>
#include <array>
#include <assert.h>

const int N = 100'000'000;

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

    std::vector<double> x(N);

    /*
     * Generate data.
     */

    // std::default_random_engine eng;
    std::minstd_rand eng;
    std::uniform_real_distribution<float> dist(-1, 1);
    #pragma omp parallel for private(eng, dist)
    for (int i = 0; i < N; i++) {
        x[i] = dist(eng);
    }

    std::vector<double> y_s(N);
    auto seq = [&]() {
        for (int i = 0; i < N; i++) {
            y_s[i] = std::sin(x[i]);
        }
    };

    double dt;
    dt = time(seq);
    std::cout << "Sequential: " << (N/dt)/1000000 << " Mops/s [" << dt << " s]" << std::endl;

    {
        std::vector<double> y_p(N);
        auto par = [&]() {
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                y_p[i] = std::sin(x[i]);
            }
        };

        dt = time(par);
        std::cout << "Parallel-for, static: " << (N/dt)/1000000 << " Mops/s [" << dt << " s]" << std::endl;

        for (int i = 0; i < N; i++) {
            assert(y_s[i] == y_p[i]);
        }
    }
    {
        std::vector<double> y_p(N);
        auto par = [&]() {
            #pragma omp parallel for schedule(dynamic, 1000)
            for (int i = 0; i < N; i++) {
                y_p[i] = std::sin(x[i]);
            }
        };

        dt = time(par);
        std::cout << "Parallel-for, dynamic: " << (N/dt)/1000000 << " Mops/s [" << dt << " s]" << std::endl;

        for (int i = 0; i < N; i++) {
            assert(y_s[i] == y_p[i]);
        }
    }
}
