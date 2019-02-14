unsigned int i = 1;

int foo() {

    for (int k = 0; k < 100; k++) {
        i = 9*i;
    }

    return i;
}