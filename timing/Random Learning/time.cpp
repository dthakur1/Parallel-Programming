#include <chrono>
#include <iostream>

int
main() {

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; i++) {
    }
    auto stop = std::chrono::steady_clock::now();
    auto dt = stop - start;
    std::cout << dt.count() << std::endl;
}
