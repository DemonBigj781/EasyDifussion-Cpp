#include "runtime.h"

#define main cosmo_sdkit_main_impl
#include COSMO_SDKIT_MAIN_SOURCE
#undef main

extern "C" int cosmo_sdkit_main(int argc, char **argv) {
    try {
        return cosmo_sdkit_main_impl(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "Diffusion server: " << error.what() << '\n';
        return 1;
    }
}
