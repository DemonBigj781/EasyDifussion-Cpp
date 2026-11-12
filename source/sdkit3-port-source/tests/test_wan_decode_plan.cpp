#include <iostream>
#include <stdexcept>

#include "model/vae/wan_decode_plan.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        for (auto version : {VERSION_WAN2, VERSION_WAN2_2_TI2V, VERSION_SD1}) {
            for (bool is_2d : {false, true}) {
                for (int tile_frames = 1; tile_frames <= 8; ++tile_frames) {
                    for (int frames = 1; frames <= 33; ++frames) {
                        auto plan = make_vae_temporal_tile_plan(frames, {tile_frames, 0});
                        const auto original = plan;
                        const bool changed = sd_version_is_wan(version) && !is_2d &&
                            plan.tiles.size() > 1 && plan.tiles.size() <= 3 && tile_frames > 2 &&
                            plan.tiles.back().end - plan.tiles.back().start == 1;
                        balance_short_wan_decode_plan(plan, version, is_2d);
                        require(plan.tiles.size() == original.tiles.size(), "tile count changed");
                        int64_t end = 0;
                        for (size_t i = 0; i < plan.tiles.size(); ++i) {
                            const auto& tile = plan.tiles[i];
                            require(tile.start == end && tile.end > tile.start, "gap, overlap or empty tile");
                            require(tile.index == original.tiles[i].index && tile.first == original.tiles[i].first &&
                                    tile.last == original.tiles[i].last && tile.overlap == 0, "tile metadata changed");
                            int64_t expected_start = original.tiles[i].start;
                            int64_t expected_end = original.tiles[i].end;
                            if (changed && i + 1 == plan.tiles.size()) --expected_start;
                            if (changed && i + 2 == plan.tiles.size()) --expected_end;
                            require(tile.start == expected_start && tile.end == expected_end, "unexpected tile adjustment");
                            end = tile.end;
                        }
                        require(end == frames, "tail frame lost");
                        if (changed) require(plan.tiles.back().end - plan.tiles.back().start == 2, "singleton tail remains");
                        sd::Tensor<float> input({1, 1, frames, 1});
                        for (int i = 0; i < frames; ++i) input.values()[i] = static_cast<float>(i);
                        auto output = process_vae_temporal_tiles(input, plan, [](const sd::Tensor<float>& tile, const VAETemporalTile&) { return tile; });
                        require(output.values() == input.values(), "tile processing lost or reordered frames");
                    }
                }
            }
        }
        std::cout << "PASS 1584 Wan decode-plan cases, singleton balancing, 2D/non-Wan/long-plan controls and frame-preserving tile processing\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
