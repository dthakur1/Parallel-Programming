#include <omp.h>
#include <iostream>

int
main() {

    int a[100];

    for (int i = 0; i < 100; i++) {
        a[i] = i;
    }

    {
        long long sum = 0;
        for (auto e : a) {
            sum += e;
        }

        std::cout << "sum (sequential): " << sum << std::endl;
    }

    {
        long long sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < 100; i++) {
            sum += a[i];
        }

        std::cout << "sum (reduction): " << sum << std::endl;
    }
}
