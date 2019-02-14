#include <omp.h>
#include <iostream>

int
main() {
    #pragma omp parallel sections
        {
            printf("common before: %d\n", omp_get_thread_num());
            #pragma omp section
                {
                printf("section 1: %d\n", omp_get_thread_num());
                }
            #pragma omp section
                {
                printf("section 2: %d\n", omp_get_thread_num());
                }
            // Get a compile-time error on the below.
            // printf("common after: %d\n", omp_get_thread_num());
        }
}
