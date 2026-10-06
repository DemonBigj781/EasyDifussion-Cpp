#define main cosmo_backward_test_impl
#include COSMO_BACKWARD_TEST_SOURCE
#undef main

extern "C" int cosmo_backward_test(void) {
    char name[] = "backward-math-test";
    char *argv[] = {name, nullptr};
    return cosmo_backward_test_impl(1, argv);
}
