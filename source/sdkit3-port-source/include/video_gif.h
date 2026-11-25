#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <functional>
#include "stable-diffusion.h"

// Portable GIF89a encoder. Fixed RGB332 palette, bounded LZW blocks.
inline std::vector<uint8_t> encode_video_gif(const sd_image_t* frames, int count, int fps,
                                             const std::function<bool()>& cancelled = {}) {
    if (!frames || count < 1 || fps < 1 || fps > 60) throw std::invalid_argument("Invalid GIF input");
    const auto w = frames[0].width, h = frames[0].height;
    if (!w || !h || w > 65535 || h > 65535) throw std::invalid_argument("Invalid GIF dimensions");
    std::vector<uint8_t> out{'G','I','F','8','9','a'};
    auto word = [&](unsigned v) { out.push_back(v & 255); out.push_back((v >> 8) & 255); };
    word(w); word(h); out.insert(out.end(), {0xf7,0,0});
    for (unsigned i=0;i<256;++i) {
        out.push_back(((i>>5)&7)*255/7); out.push_back(((i>>2)&7)*255/7); out.push_back((i&3)*255/3);
    }
    out.insert(out.end(), {0x21,0xff,11,'N','E','T','S','C','A','P','E','2','.','0',3,1,0,0,0});
    for (int f=0; f<count; ++f) {
        const auto& im=frames[f];
        if (!im.data || im.width!=w || im.height!=h || (im.channel!=1 && im.channel!=3 && im.channel!=4))
            throw std::invalid_argument("Invalid GIF frame");
        out.insert(out.end(), {0x21,0xf9,4,4});
        word(((f+1)*100+fps/2)/fps-(f*100+fps/2)/fps);
        out.insert(out.end(), {0,0,0x2c}); word(0); word(0); word(w); word(h); out.push_back(0);
        out.push_back(8);
        std::vector<uint8_t> block;
        unsigned bits=0, used=0;
        auto byte=[&](uint8_t b){block.push_back(b);if(block.size()==255){out.push_back(255);out.insert(out.end(),block.begin(),block.end());block.clear();}};
        auto code=[&](unsigned c){bits|=c<<used;used+=9;while(used>=8){byte(bits&255);bits>>=8;used-=8;}};
        // Reset before dictionary growth requires a wider code.
        code(256);
        unsigned run=0;
        for (uint64_t i=0;i<uint64_t(w)*h;++i) {
            if (i%w==0 && cancelled && cancelled()) throw std::runtime_error("GIF encoding cancelled");
            if(run==250){code(256);run=0;}
            const auto* p=im.data+i*im.channel;
            unsigned r=p[0],g=im.channel==1?r:p[1],b=im.channel==1?r:p[2];
            code((r&224)|((g&224)>>3)|(b>>6)); ++run;
        }
        code(257);if(used)byte(bits&255);
        if(!block.empty()){out.push_back(static_cast<uint8_t>(block.size()));out.insert(out.end(),block.begin(),block.end());}
        out.push_back(0);
    }
    out.push_back(0x3b);return out;
}
