#include <omp.h>
#include <stdio.h>

int
main() {
    
    #pragma omp parallel for ordered
        for (int i = 0; i < 100; i++) {
            printf("A concurrent: %d\n", i);
            #pragma omp ordered
                printf("Ordered: %d\n", i);
            printf("B concurrent: %d\n", i);
        }
}
