// Clean-room native image helpers for Easy Diffusion's draw/inpaint editor.
//
// The text-mask mode identifies compact, aligned high-contrast components and
// writes a black/white PNG suitable for diffusion inpainting.  It intentionally
// uses no code from remove_text_from_image, whose repository has no license.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace {

struct Options {
    std::string image;
    std::string output;
    float sensitivity = 0.55f;
    int padding = 3;
};

struct Box {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    int pixels = 0;
    int members = 1;
    int width() const { return x2 - x1 + 1; }
    int height() const { return y2 - y1 + 1; }
};

[[noreturn]] void usage(const char* argv0, const std::string& error = {}) {
    if (!error.empty()) std::cerr << error << "\n\n";
    std::cerr << "Usage: " << argv0
              << " text-mask --image input.png --output mask.png"
                 " [--sensitivity 0.55] [--padding 3]\n";
    std::exit(error.empty() ? 0 : 2);
}

Options parse_options(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "text-mask") usage(argv[0], "Expected text-mask mode");
    Options options;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) usage(argv[0], std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--image") options.image = value("--image");
        else if (arg == "--output") options.output = value("--output");
        else if (arg == "--sensitivity") options.sensitivity = std::stof(value("--sensitivity"));
        else if (arg == "--padding") options.padding = std::stoi(value("--padding"));
        else if (arg == "--help" || arg == "-h") usage(argv[0]);
        else usage(argv[0], "Unknown argument: " + arg);
    }
    if (options.image.empty() || options.output.empty()) usage(argv[0], "--image and --output are required");
    if (!(options.sensitivity >= 0.f && options.sensitivity <= 1.f))
        usage(argv[0], "--sensitivity must be in [0,1]");
    if (options.padding < 0 || options.padding > 64) usage(argv[0], "--padding must be in [0,64]");
    return options;
}

std::vector<uint8_t> dilate(const std::vector<uint8_t>& input, int width, int height, int rx, int ry) {
    std::vector<uint8_t> horizontal(input.size(), 0), output(input.size(), 0);
    for (int y = 0; y < height; ++y) {
        int active = 0;
        for (int x = -rx; x < width + rx; ++x) {
            const int add = x + rx;
            const int sub = x - rx - 1;
            if (add >= 0 && add < width) active += input[static_cast<size_t>(y) * width + add] != 0;
            if (sub >= 0 && sub < width) active -= input[static_cast<size_t>(y) * width + sub] != 0;
            if (x >= 0 && x < width) horizontal[static_cast<size_t>(y) * width + x] = active > 0 ? 255 : 0;
        }
    }
    for (int x = 0; x < width; ++x) {
        int active = 0;
        for (int y = -ry; y < height + ry; ++y) {
            const int add = y + ry;
            const int sub = y - ry - 1;
            if (add >= 0 && add < height) active += horizontal[static_cast<size_t>(add) * width + x] != 0;
            if (sub >= 0 && sub < height) active -= horizontal[static_cast<size_t>(sub) * width + x] != 0;
            if (y >= 0 && y < height) output[static_cast<size_t>(y) * width + x] = active > 0 ? 255 : 0;
        }
    }
    return output;
}

std::vector<Box> components(const std::vector<uint8_t>& mask, int width, int height) {
    std::vector<uint8_t> seen(mask.size(), 0);
    std::vector<Box> result;
    std::vector<int> pending;
    pending.reserve(4096);
    constexpr int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    constexpr int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int start = y * width + x;
            if (!mask[start] || seen[start]) continue;
            Box box{x, y, x, y, 0, 1};
            pending.clear();
            pending.push_back(start);
            seen[start] = 1;
            while (!pending.empty()) {
                const int index = pending.back();
                pending.pop_back();
                const int px = index % width, py = index / width;
                box.x1 = std::min(box.x1, px); box.x2 = std::max(box.x2, px);
                box.y1 = std::min(box.y1, py); box.y2 = std::max(box.y2, py);
                ++box.pixels;
                for (int d = 0; d < 8; ++d) {
                    const int nx = px + dx[d], ny = py + dy[d];
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                    const int next = ny * width + nx;
                    if (mask[next] && !seen[next]) {
                        seen[next] = 1;
                        pending.push_back(next);
                    }
                }
            }
            result.push_back(box);
        }
    }
    return result;
}

bool same_text_line(const Box& a, const Box& b) {
    const float ah = static_cast<float>(a.height()), bh = static_cast<float>(b.height());
    const float height_ratio = std::min(ah, bh) / std::max(ah, bh);
    if (height_ratio < 0.28f) return false;
    const int overlap = std::min(a.y2, b.y2) - std::max(a.y1, b.y1) + 1;
    const float center_a = (a.y1 + a.y2) * .5f, center_b = (b.y1 + b.y2) * .5f;
    const bool aligned = overlap > 0 || std::abs(center_a - center_b) <= .55f * std::max(ah, bh);
    const int gap = std::max({b.x1 - a.x2 - 1, a.x1 - b.x2 - 1, 0});
    return aligned && gap <= static_cast<int>(3.5f * std::max(ah, bh));
}

std::vector<uint8_t> make_text_mask(const uint8_t* rgb, int width, int height,
                                    const Options& options, int* regions_out) {
    const size_t size = static_cast<size_t>(width) * height;
    std::vector<uint8_t> gray(size), edge(size, 0);
    for (size_t i = 0; i < size; ++i) {
        gray[i] = static_cast<uint8_t>((77 * rgb[3 * i] + 150 * rgb[3 * i + 1] + 29 * rgb[3 * i + 2]) >> 8);
    }

    std::vector<int> strengths;
    strengths.reserve(size / 3);
    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            const int gx = -gray[i - width - 1] - 2 * gray[i - 1] - gray[i + width - 1]
                           + gray[i - width + 1] + 2 * gray[i + 1] + gray[i + width + 1];
            const int gy = -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1]
                           + gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
            strengths.push_back(std::min(1020, std::abs(gx) + std::abs(gy)));
        }
    }
    if (strengths.empty()) throw std::runtime_error("Image is too small");
    const float percentile = 0.88f - options.sensitivity * 0.28f;
    const size_t rank = std::min(strengths.size() - 1,
                                 static_cast<size_t>(percentile * strengths.size()));
    std::nth_element(strengths.begin(), strengths.begin() + rank, strengths.end());
    const int threshold = std::clamp(strengths[rank], 48, 360);

    for (int y = 1; y + 1 < height; ++y) {
        for (int x = 1; x + 1 < width; ++x) {
            const size_t i = static_cast<size_t>(y) * width + x;
            const int gx = -gray[i - width - 1] - 2 * gray[i - 1] - gray[i + width - 1]
                           + gray[i - width + 1] + 2 * gray[i + 1] + gray[i + width + 1];
            const int gy = -gray[i - width - 1] - 2 * gray[i - width] - gray[i - width + 1]
                           + gray[i + width - 1] + 2 * gray[i + width] + gray[i + width + 1];
            edge[i] = std::abs(gx) + std::abs(gy) >= threshold ? 255 : 0;
        }
    }

    const auto joined = dilate(edge, width, height, 1, 1);
    auto glyphs = components(joined, width, height);
    glyphs.erase(std::remove_if(glyphs.begin(), glyphs.end(), [&](const Box& b) {
        const int area = b.width() * b.height();
        const float density = static_cast<float>(b.pixels) / std::max(area, 1);
        return b.width() < 2 || b.height() < 3 || b.width() > width * .18f || b.height() > height * .16f
               || area < 12 || density < .08f || density > .98f
               || b.width() > b.height() * 8.f || b.height() > b.width() * 12.f;
    }), glyphs.end());

    std::sort(glyphs.begin(), glyphs.end(), [](const Box& a, const Box& b) {
        return a.x1 == b.x1 ? a.y1 < b.y1 : a.x1 < b.x1;
    });
    std::vector<Box> lines;
    for (const Box& glyph : glyphs) {
        int best = -1, best_gap = width + height;
        for (int i = 0; i < static_cast<int>(lines.size()); ++i) {
            if (!same_text_line(lines[i], glyph)) continue;
            const int gap = std::max({glyph.x1 - lines[i].x2 - 1, lines[i].x1 - glyph.x2 - 1, 0});
            if (gap < best_gap) { best = i; best_gap = gap; }
        }
        if (best < 0) {
            lines.push_back(glyph);
        } else {
            Box& line = lines[best];
            line.x1 = std::min(line.x1, glyph.x1); line.y1 = std::min(line.y1, glyph.y1);
            line.x2 = std::max(line.x2, glyph.x2); line.y2 = std::max(line.y2, glyph.y2);
            line.pixels += glyph.pixels;
            line.members += glyph.members;
        }
    }

    lines.erase(std::remove_if(lines.begin(), lines.end(), [&](const Box& b) {
        const float aspect = static_cast<float>(b.width()) / b.height();
        return b.members < 2 || b.width() < 7 || aspect < .65f || b.height() > height * .22f;
    }), lines.end());

    std::vector<uint8_t> selected(size, 0);
    for (const Box& line : lines) {
        const int pad = std::max(options.padding, static_cast<int>(std::round(line.height() * .08f)));
        const int x1 = std::max(0, line.x1 - pad), x2 = std::min(width - 1, line.x2 + pad);
        const int y1 = std::max(0, line.y1 - pad), y2 = std::min(height - 1, line.y2 + pad);
        for (int y = y1; y <= y2; ++y) {
            for (int x = x1; x <= x2; ++x) {
                const size_t i = static_cast<size_t>(y) * width + x;
                if (edge[i]) selected[i] = 255;
            }
        }
    }
    *regions_out = static_cast<int>(lines.size());
    return dilate(selected, width, height, std::max(1, options.padding), std::max(1, options.padding));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int width = 0, height = 0, channels = 0;
        uint8_t* image = stbi_load(options.image.c_str(), &width, &height, &channels, 3);
        if (!image || width <= 2 || height <= 2) {
            stbi_image_free(image);
            throw std::runtime_error("Could not decode input image: " + options.image);
        }
        int regions = 0;
        auto mask = make_text_mask(image, width, height, options, &regions);
        stbi_image_free(image);
        if (!stbi_write_png(options.output.c_str(), width, height, 1, mask.data(), width))
            throw std::runtime_error("Could not write output mask: " + options.output);
        std::cout << "{\"width\":" << width << ",\"height\":" << height
                  << ",\"regions\":" << regions << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sdkit-image-tools: " << error.what() << "\n";
        return 1;
    }
}
