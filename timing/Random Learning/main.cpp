#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

int
main() {

    std::default_random_engine eng;
    std::uniform_real_distribution d;

    std::vector<double> a;
    for (std::size_t i = 0; i < 100'000'000; i++) {
        a.push_back(d(eng));
    }
    auto start = std::chrono::steady_clock::now();
    std::sort(a.begin(), a.end());
    auto stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = stop - start;
    std::cout << dt.count() << std::endl;
}
