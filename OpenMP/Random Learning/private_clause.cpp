#include <omp.h>
#include <iostream>

class A {
    public:
        A() {
            std::cout << "A::A()" << std::endl;
        }
};

int
main(int argc, char *argv[])  {

    A a;

    #pragma omp parallel private(a)
        ;
}
