#define main cosmo_optimizer_test_impl
#include COSMO_OPTIMIZER_TEST_SOURCE
#undef main

extern "C" int cosmo_optimizer_test(void) {
    try {
        return cosmo_optimizer_test_impl();
    } catch (const std::exception &error) {
        std::cerr << "Optimizer test: " << error.what() << '\n';
        return 1;
    }
}
