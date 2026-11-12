#ifndef __SD_MODEL_VAE_WAN_DECODE_PLAN_H__
#define __SD_MODEL_VAE_WAN_DECODE_PLAN_H__

#include "model.h"
#include "model/vae/vae_tiling.hpp"

static inline void balance_short_wan_decode_plan(VAETemporalTilePlan& plan, SDVersion version, bool is_2d) {
    if (sd_version_is_wan(version) && !is_2d && plan.tiles.size() > 1 &&
        plan.tiles.size() <= 3 && plan.tile_frames > 2 &&
        plan.tiles.back().end - plan.tiles.back().start == 1) {
        // Longer plans still contain full middle tiles, so moving their tail does not reduce the peak.
        auto& previous = plan.tiles[plan.tiles.size() - 2];
        auto& last = plan.tiles.back();
        --previous.end;
        --last.start;
    }
}

#endif  // __SD_MODEL_VAE_WAN_DECODE_PLAN_H__
