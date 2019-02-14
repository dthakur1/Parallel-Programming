#ifndef GET_TSC_HPP
#define GET_TSC_HPP

inline unsigned long long
get_tsc() {
    unsigned int a, d;
    // asm volatile("lfence; rdtsc; lfence" : "=a" (a), "=d" (d));
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return (unsigned long) a | (((unsigned long) d) << 32);;
}

#endif
