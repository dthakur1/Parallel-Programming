#include <stdio.h>
#include <assert.h>
#include <type_traits>
#include <iostream>

class A {
    public:
        __device__ __host__ A() : i(0), x(3.14) {}
        __device__ __host__ void foo() {
            printf("Hello!\n");
        }
    private:
        int i;
        double x;
};


__global__ void kernel() {
    A a;
    a.foo();
}


int
main() {

    std::cout << std::boolalpha << std::is_trivially_copyable<A>::value << std::endl;

    /*
    kernel<<<1, 32>>>();
    assert(cudaPeekAtLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    */
}
