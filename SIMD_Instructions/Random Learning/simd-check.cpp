#include <iostream>

#define check(s) \
    do { \
        if (__builtin_cpu_supports(s)) { \
            std::cout << s << std::endl; \
        } \
    } while (0)

int
main() {
    check("mmx");
    check("sse");
    check("sse4.2");
    check("avx");
    check("avx2");
}
