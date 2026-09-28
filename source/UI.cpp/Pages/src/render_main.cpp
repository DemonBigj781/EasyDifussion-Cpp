#include "easy_diffusion/ui/pages.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: easy-diffusion-ui-render PAGE_PATH\n";
        return 2;
    }
    try {
        std::cout << easy_diffusion::ui::pages::render(argv[1]).render();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
