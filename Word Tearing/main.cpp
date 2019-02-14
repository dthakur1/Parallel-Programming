#include <thread>
#include <cstdio>
#include <cassert>

unsigned int a[100];
volatile unsigned int *ip;

void
writer() {

    unsigned short i = 0;
    while (true) {
        unsigned int val = (unsigned int) i | ((unsigned int) i << 16);
        *ip = val;
        i++;
    }
}

void
reader() {

    unsigned long iter = 0;
    while (true) {
        unsigned int val = *ip;
        unsigned int low = 0xffff&val;
        unsigned int high = val >> 16;
        if (low != high) {
            printf("ERROR: %lu: low=%u, high=%u\n", iter, low, high);
        }
        if (iter%10'000'000 == 0) {
            printf("%lu: low=%u, high=%u\n", iter, low, high);
        }
        iter++;
    }
}

int
main() {

    // ip = (unsigned int *) a;
    ip = (unsigned int *) ((((unsigned long) a + 63)&~63UL) + 62);

    assert((unsigned long) ip >= (unsigned long) a
     && (unsigned long) ip < (unsigned long) a + 100);

    /*
    printf("%lu, %lu : %lu, %lu\n",
     (unsigned long) a,
     (unsigned long) a % 64,
     (unsigned long) ip,
     (unsigned long) ip % 64);
    */
    
    std::thread writer_thread(writer);
    std::thread reader_thread(reader);

    writer_thread.join();
    reader_thread.join();
}
