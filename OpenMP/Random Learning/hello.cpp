#include <omp.h>
#include <stdio.h>
int
main() {
    int nthreads, tid;
    // Spawn a team of threads.
    #pragma omp parallel private(tid)
        {
            // Get thread ID.
            tid = omp_get_thread_num();
            printf("Hello World from thread = %d\n", tid);
            // Only master thread.
            if (tid == 0) {
                nthreads = omp_get_num_threads();
                printf("Number of threads = %d\n", nthreads);
            }
        } // All threads join master.
}
