#include <omp.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>

int
main() {

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid%2 == 0) {
            sleep(1);
        }
        printf("Before, thread %d\n", tid);
        #pragma omp barrier
            std::cout << "After thread " << tid << std::endl;
    }
}