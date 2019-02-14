#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int
main() {

    #pragma omp parallel
    {
        // #pragma omp single nowait
        {
            for (int i = 0; i < 10; i++) {
                // #pragma omp task firstprivate(i)
                // #pragma omp task private(i)
                #pragma omp task
                {
                    sleep(1);
                    printf("iteration %d\n", i);
                }
            }
            printf("Before taskwait\n");
            #pragma omp taskwait
            printf("After taskwait\n");
        }
    }
}
