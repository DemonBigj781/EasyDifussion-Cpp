#include "runtime.h"
#include "easy_diffusion/ui/pages.hpp"

#include <cstdio>
#include <exception>
#include <string>

extern "C" int cosmo_ui_render(const char *path) {
    try {
        const std::string html = easy_diffusion::ui::pages::render(path).render();
        return std::fwrite(html.data(), 1, html.size(), stdout) == html.size() ? 0 : 1;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "UI render: %s\n", error.what());
        return 1;
    }
}

extern "C" int cosmo_ui_selftest(void) {
    try {
        const auto &pages = easy_diffusion::ui::pages::definitions();
        if (pages.empty()) return 1;
        for (const auto &page : pages) {
            const std::string html = easy_diffusion::ui::pages::render(page.path).render();
            if (html.find("id=\"page-content\"") == std::string::npos ||
                html.find("/cpp-ui/assets/ui.css") == std::string::npos) {
                std::fprintf(stderr, "UI page failed: %s\n", page.path.c_str());
                return 1;
            }
        }
        FILE *asset = std::fopen("/zip/cpp-ui/assets/ui.css", "rb");
        if (!asset) {
            std::perror("Embedded UI stylesheet");
            return 1;
        }
        const int first = std::fgetc(asset);
        std::fclose(asset);
        if (first == EOF) return 1;
        std::printf("UI_SELFTEST pages=%zu embedded_assets=yes PASS\n", pages.size());
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "UI self-test: %s\n", error.what());
        return 1;
    }
}
