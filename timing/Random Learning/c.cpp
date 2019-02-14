#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

double time(int n) {

    std::default_random_engine eng;
    std::uniform_real_distribution d;

    std::vector<double> a;
    for (int i = 0; i < n; i++) {
        a.push_back(d(eng));
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(a.begin(), a.end());
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = stop - start;
    return dt.count();
}

int
main() {

    std::cout << time(10) << std::endl;
    std::cout << time(100) << std::endl;
    std::cout << time(1000) << std::endl;
    std::cout << time(10000) << std::endl;
}
