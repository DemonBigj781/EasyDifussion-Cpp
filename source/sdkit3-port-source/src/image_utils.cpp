#include "image_utils.h"

#include <cctype>
#include <algorithm>
#include <new>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <initializer_list>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "../stable-diffusion.cpp/thirdparty/stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../stable-diffusion.cpp/thirdparty/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../stable-diffusion.cpp/thirdparty/stb_image_resize.h"
#include "base64.hpp"
#include "logging.h"

namespace {

// The bundled stb interfaces and their row offsets use signed int arithmetic.
// Validate before narrowing sd_image_t's unsigned fields or multiplying sizes.
bool checked_image_bytes(uint32_t width, uint32_t height, uint32_t channels, size_t& bytes) {
    const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max());
    if (!width || !height || channels < 1 || channels > 4 ||
        width > limit || height > limit || width > limit / channels) return false;
    const size_t stride = static_cast<size_t>(width) * channels;
    if (height > limit / stride) return false;
    bytes = stride * height;
    return true;
}

// Mirror the bundled stb default-filter workspace layout in checked arithmetic.
// Raw pixel bounds alone do not cover coefficient, decode or ring buffers.
// Keep this beside the included implementation and review when stb is updated.
bool checked_resize_workspace(int input_w, int input_h, int output_w, int output_h, int channels) {
    const uint64_t limit = static_cast<uint64_t>(std::numeric_limits<int>::max());
    struct Axis {
        uint64_t pixels;
        uint64_t coefficients;
        uint64_t contributors;
        bool up;
    };
    auto axis = [limit](int input, int output, Axis& result) {
        const float scale = static_cast<float>(output) / input;
        result.up = scale > 1;
        const auto filter = result.up ? STBIR_DEFAULT_FILTER_UPSAMPLE : STBIR_DEFAULT_FILTER_DOWNSAMPLE;
        const float support = stbir__filter_info_table[filter].support(result.up ? 1 / scale : scale);
        // Preserve stb's float arithmetic before ceil, but check before casts.
        const double pixels = std::ceil(result.up ? support * 2 : support * 2 / scale);
        const double coefficients = std::ceil(support * 2);
        if (!std::isfinite(pixels) || !std::isfinite(coefficients) ||
            pixels < 1 || coefficients < 1 || pixels > limit || coefficients > limit) return false;
        result.pixels = static_cast<uint64_t>(pixels);
        result.coefficients = static_cast<uint64_t>(coefficients);
        result.contributors = result.up ? static_cast<uint64_t>(output)
            : static_cast<uint64_t>(input) + (result.pixels / 2) * 2;
        return result.contributors <= limit;
    };
    Axis horizontal{}, vertical{};
    if (!axis(input_w, output_w, horizontal) || !axis(input_h, output_h, vertical)) return false;
    auto product = [limit](uint64_t& value, std::initializer_list<uint64_t> factors) {
        value = 1;
        for (auto factor : factors) {
            if (factor && value > limit / factor) return false;
            value *= factor;
        }
        return true;
    };
    uint64_t sizes[8]{};
    if (!product(sizes[0], {horizontal.contributors, sizeof(stbir__contributors)}) ||
        !product(sizes[1], {horizontal.contributors, horizontal.coefficients, sizeof(float)}) ||
        !product(sizes[2], {vertical.contributors, sizeof(stbir__contributors)}) ||
        !product(sizes[3], {vertical.contributors, vertical.coefficients, sizeof(float)}) ||
        !product(sizes[4], {static_cast<uint64_t>(input_w) + (horizontal.pixels / 2) * 2,
                            static_cast<uint64_t>(channels), sizeof(float)}) ||
        !product(sizes[5], {static_cast<uint64_t>(output_w), static_cast<uint64_t>(channels), sizeof(float)}) ||
        !product(sizes[6], {static_cast<uint64_t>(output_w), static_cast<uint64_t>(channels),
                            vertical.pixels, sizeof(float)}) ||
        !product(sizes[7], {static_cast<uint64_t>(output_w), static_cast<uint64_t>(channels), sizeof(float)})) return false;
    // stb computes both fields before zeroing the unused horizontal/encode one.
    sizes[vertical.up ? 5 : 7] = 0;
    uint64_t total = 0;
    for (auto size : sizes) {
        if (total > limit - size) return false;
        total += size;
    }
    return true;
}

bool starts_with_data_uri(const std::string& value) {
    static constexpr char prefix[] = "data:";
    if (value.size() < sizeof(prefix) - 1) {
        return false;
    }
    for (size_t index = 0; index < sizeof(prefix) - 1; ++index) {
        if (std::tolower(static_cast<unsigned char>(value[index])) != prefix[index]) {
            return false;
        }
    }
    return true;
}

bool normalizeBase64Image(const std::string& input, std::string& output) {
    size_t begin = 0;
    size_t end = input.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }

    std::string value = input.substr(begin, end - begin);
    if (starts_with_data_uri(value)) {
        const size_t comma = value.find(',');
        if (comma == std::string::npos) {
            LOG_ERROR("Invalid image data URI: missing payload separator");
            return false;
        }
        std::string header = value.substr(0, comma);
        for (char& character : header) {
            character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        }
        if (header.find(";base64") == std::string::npos) {
            LOG_ERROR("Invalid image data URI: payload is not base64 encoded");
            return false;
        }
        value.erase(0, comma + 1);
    }

    output.clear();
    output.reserve(value.size() + 3);
    bool saw_padding = false;
    size_t padding = 0;
    for (const unsigned char character : value) {
        if (std::isspace(character)) {
            continue;
        }
        if (character == '=') {
            saw_padding = true;
            ++padding;
            output.push_back('=');
            continue;
        }
        const bool alphabet =
            (character >= 'A' && character <= 'Z') ||
            (character >= 'a' && character <= 'z') ||
            (character >= '0' && character <= '9') || character == '+' || character == '/';
        if (!alphabet || saw_padding) {
            LOG_ERROR("Invalid base64 image payload (length: %zu)", value.size());
            return false;
        }
        output.push_back(static_cast<char>(character));
    }

    if (output.empty() || padding > 2 || output.size() % 4 == 1) {
        LOG_ERROR("Invalid base64 image payload length: %zu", output.size());
        return false;
    }
    if (padding == 0) {
        output.append((4 - output.size() % 4) % 4, '=');
    } else if (output.size() % 4 != 0) {
        LOG_ERROR("Invalid padded base64 image payload length: %zu", output.size());
        return false;
    }
    return true;
}

}  // namespace

std::string imageToBase64(const sd_image_t& image) {
    size_t bytes = 0;
    if (!image.data || !checked_image_bytes(image.width, image.height, image.channel, bytes)) {
        return "";
    }
    const size_t stride_bytes = static_cast<size_t>(image.width) * image.channel;
    // PNG adds one filter byte to every row before passing an int length to zlib.
    if (image.height > static_cast<size_t>(std::numeric_limits<int>::max()) / (stride_bytes + 1)) {
        return "";
    }

    // Use stb_image_write to convert to PNG format
    std::vector<unsigned char> png_buffer;

    // Lambda function to capture PNG data
    auto write_callback = [](void* context, void* data, int size) {
        auto* buffer = static_cast<std::vector<unsigned char>*>(context);
        unsigned char* bytes = static_cast<unsigned char*>(data);
        buffer->insert(buffer->end(), bytes, bytes + size);
    };

    // Write PNG to buffer (stride is width * channels for tightly packed data)
    int stride = static_cast<int>(stride_bytes);
    int result = stbi_write_png_to_func(write_callback, &png_buffer, image.width, image.height, image.channel,
                                        image.data, stride);

    if (result == 0 || png_buffer.empty()) {
        LOG_ERROR("Failed to encode image as PNG");
        return "";
    }

    // Encode PNG data as base64
    return base64_encode(png_buffer.data(), png_buffer.size());
}

sd_image_t base64ToImage(const std::string& base64_data, int desired_channels) {
    sd_image_t image = {0, 0, 0, nullptr};

    if (desired_channels < 0 || desired_channels > 4 || base64_data.empty()) {
        return image;
    }

    std::string base64_clean;
    if (!normalizeBase64Image(base64_data, base64_clean)) {
        return image;
    }

    // Decode base64
    std::vector<unsigned char> decoded_data = base64_decode(base64_clean);

    if (decoded_data.empty()) {
        LOG_ERROR("Failed to decode base64 data");
        return image;
    }

    if (decoded_data.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        LOG_ERROR("Encoded image exceeds the decoder's supported byte range");
        return image;
    }

    // Load image using stb_image
    int width, height, channels;
    unsigned char* img_data =
        stbi_load_from_memory(decoded_data.data(), static_cast<int>(decoded_data.size()), &width, &height, &channels, desired_channels);

    if (!img_data) {
        LOG_ERROR("Failed to load image from memory: %s", stbi_failure_reason());
        return image;
    }

    // Use the desired channels if specified, otherwise use the actual channels
    int final_channels = desired_channels > 0 ? desired_channels : channels;

    size_t bytes = 0;
    if (width <= 0 || height <= 0 || !checked_image_bytes(width, height, final_channels, bytes)) {
        stbi_image_free(img_data);
        LOG_ERROR("Decoded image dimensions exceed the supported byte range");
        return image;
    }

    image.width = width;
    image.height = height;
    image.channel = final_channels;
    image.data = img_data;

    LOG_DEBUG("Decoded image: %dx%d, %d channels", width, height, final_channels);

    return image;
}

sd_image_t createSolidImage(int width, int height, int channels, uint8_t value) {
    sd_image_t image = {0, 0, 0, nullptr};
    size_t bytes = 0;
    if (width <= 0 || height <= 0 || channels <= 0 ||
        !checked_image_bytes(width, height, channels, bytes)) return image;
    auto* data = static_cast<uint8_t*>(malloc(bytes));
    if (!data) return image;
    memset(data, value, bytes);
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.channel = static_cast<uint32_t>(channels);
    image.data = data;
    return image;
}

void freeImage(sd_image_t& image) {
    if (image.data) {
        stbi_image_free(image.data);
        image.data = nullptr;
    }
    image.width = 0;
    image.height = 0;
    image.channel = 0;
}

bool resizeImage(sd_image_t& image, int target_width, int target_height, bool clamp_to_8) {
    size_t source_bytes = 0;
    if (!image.data || !checked_image_bytes(image.width, image.height, image.channel, source_bytes)) {
        LOG_ERROR("Cannot resize invalid image");
        return false;
    }

    if (target_width <= 0 || target_height <= 0) {
        LOG_ERROR("Invalid target dimensions: %dx%d", target_width, target_height);
        return false;
    }

    // Clamp to multiples of 8 if requested
    int final_width = target_width;
    int final_height = target_height;
    if (clamp_to_8) {
        final_width = target_width - (target_width % 8);
        final_height = target_height - (target_height % 8);
        if (final_width <= 0) final_width = 8;
        if (final_height <= 0) final_height = 8;
    }

    size_t new_size = 0;
    if (!checked_image_bytes(final_width, final_height, image.channel, new_size)) {
        LOG_ERROR("Target image dimensions exceed the supported byte range");
        return false;
    }

    if (image.width == final_width && image.height == final_height) {
        // Already at target size
        return true;
    }

    if (!checked_resize_workspace(static_cast<int>(image.width), static_cast<int>(image.height),
                                  final_width, final_height, static_cast<int>(image.channel))) {
        LOG_ERROR("Resize workspace exceeds the supported integer range");
        return false;
    }

    // Allocate new buffer for resized image
    unsigned char* resized_data = (unsigned char*)malloc(new_size);
    if (!resized_data) {
        LOG_ERROR("Failed to allocate memory for resized image");
        return false;
    }

    // Resize using stb_image_resize
    int result = stbir_resize_uint8(image.data, image.width, image.height, 0, resized_data, final_width, final_height,
                                    0, image.channel);

    if (result == 0) {
        LOG_ERROR("Failed to resize image");
        free(resized_data);
        return false;
    }

    // Free old data and update image
    stbi_image_free(image.data);
    image.data = resized_data;
    image.width = final_width;
    image.height = final_height;

    LOG_DEBUG("Resized image to %dx%d", final_width, final_height);
    return true;
}

bool blurMaskImage(sd_image_t& image, float sigma) {
    size_t pixels = 0;
    if (!image.data || image.channel != 1 ||
        !checked_image_bytes(image.width, image.height, image.channel, pixels) ||
        !std::isfinite(sigma) || sigma < 0 || sigma > kMaxMaskBlur) return false;
    if (sigma == 0) return true;
    try {
        const int radius = std::max(1, static_cast<int>(std::ceil(2.5 * sigma)));
        std::vector<double> kernel(2 * radius + 1);
        double sum = 0;
        for (int k = -radius; k <= radius; ++k) {
            const double scaled = static_cast<double>(k) / sigma;
            kernel[k + radius] = std::exp(-0.5 * scaled * scaled);
            sum += kernel[k + radius];
        }
        for (auto& value : kernel) value /= sum;
        std::vector<float> horizontal(pixels);
        const int64_t width = image.width, height = image.height;
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                double value = 0;
                for (int k = -radius; k <= radius; ++k) {
                    const auto sx = std::clamp<int64_t>(x + k, 0, width - 1);
                    value += image.data[y * width + sx] * kernel[k + radius];
                }
                horizontal[y * width + x] = static_cast<float>(value);
            }
        }
        // All allocations and the horizontal read complete before mutating input.
        for (int64_t y = 0; y < height; ++y) {
            for (int64_t x = 0; x < width; ++x) {
                double value = 0;
                for (int k = -radius; k <= radius; ++k) {
                    const auto sy = std::clamp<int64_t>(y + k, 0, height - 1);
                    value += horizontal[sy * width + x] * kernel[k + radius];
                }
                image.data[y * width + x] = static_cast<uint8_t>(std::clamp(value, 0.0, 255.0) + 0.5);
            }
        }
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    } catch (const std::length_error&) {
        return false;
    }
}

sd_image_t prepareMaskImage(const std::string& encoded_mask, int width, int height,
                            float sigma, bool invert) {
    sd_image_t mask_image = {0, 0, 0, nullptr};

    if (encoded_mask.empty()) {
        // Create a default mask for img2img (all white = no masking)
        mask_image = createSolidImage(width, height, 1, 255);
        if (!mask_image.data) {
            throw std::runtime_error("Failed to allocate mask image: invalid dimensions or insufficient memory");
        }

        return mask_image;
    }

    // Decode provided mask
    mask_image = base64ToImage(encoded_mask, 1);
    if (!mask_image.data) {
        throw std::runtime_error("Failed to decode mask image");
    }

    // Resize mask to match init image dimensions if needed
    if (mask_image.width != width || mask_image.height != height) {
        if (!resizeImage(mask_image, width, height, true)) {
            freeImage(mask_image);
            throw std::runtime_error("Failed to resize mask image");
        }
    }

    if (invert) {
        const size_t pixels = static_cast<size_t>(mask_image.width) * mask_image.height;
        for (size_t i = 0; i < pixels; ++i) mask_image.data[i] = 255 - mask_image.data[i];
    }
    if (!blurMaskImage(mask_image, sigma)) {
        freeImage(mask_image);
        throw std::runtime_error("Failed to blur mask image");
    }
    return mask_image;
}

namespace {
// Advanced conditioning follows the established adapter's RGB Lanczos / mask
// bilinear / premultiplied RGBA bilinear contract. It does not replace ordinary
// img2img's existing resize implementation. Separable byte rounding is explicit.
// Bound untrusted image headers before stb allocates decoded pixels. Ordinary
// img2img keeps its existing contract; these stricter caps apply only here.
sd_image_t decode_advanced_image(const std::string& encoded, int channels) {
    constexpr size_t encoded_limit = 24u * 1024u * 1024u;
    if (encoded.empty() || encoded.size() > encoded_limit)
        throw std::invalid_argument("Advanced Inpaint encoded image exceeds 24 MiB or is empty");
    std::string normalized;
    if (!normalizeBase64Image(encoded, normalized))
        throw std::invalid_argument("Advanced Inpaint image encoding is invalid");
    auto bytes = base64_decode(normalized);
    int w=0,h=0,c=0;
    if (bytes.empty() || bytes.size() > std::numeric_limits<int>::max() ||
        !stbi_info_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &c) ||
        w <= 0 || h <= 0 || w > 16384 || h > 16384 || static_cast<uint64_t>(w)*h > 16777216)
        throw std::invalid_argument("Advanced Inpaint image header is invalid or exceeds 16 megapixels");
    return base64ToImage(encoded, channels);
}
struct AdvancedImage {
    sd_image_t image{0, 0, 0, nullptr};
    ~AdvancedImage() { freeImage(image); }
};
double advanced_kernel(double x, bool lanczos) {
    x = std::abs(x);
    if (!lanczos) return x < 1 ? 1 - x : 0;
    if (x == 0) return 1;
    if (x >= 3) return 0;
    const double pi = 3.14159265358979323846;
    return std::sin(pi*x)*std::sin(pi*x/3)/(pi*pi*x*x/3);
}
void advanced_resize(sd_image_t& image, int width, int height, bool lanczos) {
    if (image.width == static_cast<unsigned>(width) && image.height == static_cast<unsigned>(height)) return;
    // A bounded temporary image and filter table; reject before allocation.
    size_t bytes;
    if (width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
        !checked_image_bytes(width, height, image.channel, bytes) ||
        static_cast<uint64_t>(width)*height > 16777216 ||
        static_cast<uint64_t>(width)*image.height > 16777216)
        throw std::invalid_argument("Advanced Inpaint dimensions exceed bounded resize limits");
    auto axis = [lanczos](const sd_image_t& input, sd_image_t& output, bool horizontal) {
        const int n = horizontal ? input.width : input.height;
        const int target = horizontal ? output.width : output.height;
        const double scale = static_cast<double>(n)/target;
        const double filter_scale = std::max(1., scale), support = (lanczos ? 3 : 1)*filter_scale;
        for (int p=0; p<target; ++p) {
            const double center = (p+.5)*scale;
            const int lo = std::max(0, static_cast<int>(center-support+.5));
            const int hi = std::min(n, static_cast<int>(center+support+.5));
            std::vector<double> weights(hi-lo); double sum=0;
            for (int i=lo;i<hi;++i) sum += weights[i-lo]=advanced_kernel((i-center+.5)/filter_scale,lanczos);
            std::vector<int> fixed(hi-lo);
            for (int i=lo;i<hi;++i) {
                const double weight=weights[i-lo]/sum;
                fixed[i-lo]=static_cast<int>(weight*(1<<22)+(weight<0 ? -.5 : .5));
            }
            const int lines=horizontal ? input.height : input.width;
            for (int line=0;line<lines;++line) for (unsigned c=0;c<input.channel;++c) {
                int64_t value=1<<21;
                for (int i=lo;i<hi;++i) {
                    size_t at=horizontal ? (static_cast<size_t>(line)*input.width+i)*input.channel+c
                                         : (static_cast<size_t>(i)*input.width+line)*input.channel+c;
                    value += static_cast<int64_t>(input.data[at])*fixed[i-lo];
                }
                size_t at=horizontal ? (static_cast<size_t>(line)*output.width+p)*output.channel+c
                                     : (static_cast<size_t>(p)*output.width+line)*output.channel+c;
                output.data[at]=static_cast<uint8_t>(std::max<int64_t>(0,std::min<int64_t>(255,value>>22)));
            }
        }
    };
    AdvancedImage horizontal, result;
    horizontal.image=createSolidImage(width,image.height,image.channel,0);
    result.image=createSolidImage(width,height,image.channel,0);
    if (!horizontal.image.data || !result.image.data) throw std::bad_alloc();
    axis(image,horizontal.image,true);axis(horizontal.image,result.image,false);
    freeImage(image);image=result.image;result.image.data=nullptr;
}
}

void prepareAdvancedInpaint(const std::string& encoded_source, const std::string& encoded_guide,
                            const std::string& encoded_mask, int width, int height,
                            float strength, float sigma, bool invert,
                            sd_image_t& conditioned, sd_image_t& effective_mask) {
    if (&conditioned == &effective_mask || conditioned.data || effective_mask.data) throw std::invalid_argument("Advanced Inpaint outputs must be empty");
    if (!std::isfinite(strength) || strength <= 0 || strength >= 1)
        throw std::invalid_argument("Advanced Inpaint requires denoising strength strictly between 0 and 1");
    if (width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
        static_cast<uint64_t>(width)*height > 16777216)
        throw std::invalid_argument("Advanced Inpaint dimensions exceed bounded limits");
    AdvancedImage base, guide, mask;
    base.image=decode_advanced_image(encoded_source,3);guide.image=decode_advanced_image(encoded_guide,0);
    if (!base.image.data) throw std::invalid_argument("Advanced Inpaint requires a decodable original image");
    if (!guide.image.data || (guide.image.channel != 2 && guide.image.channel != 4))
        throw std::invalid_argument("Advanced Inpaint requires a guide with an alpha channel");
    if (base.image.width!=guide.image.width || base.image.height!=guide.image.height)
        throw std::invalid_argument("Advanced Inpaint guide and original dimensions must match");
    const size_t pixels=static_cast<size_t>(guide.image.width)*guide.image.height;
    bool painted=false;
    for(size_t i=0;i<pixels;++i) painted |= guide.image.data[i*guide.image.channel+guide.image.channel-1]!=0;
    if(!painted) throw std::invalid_argument("Advanced Inpaint guide contains no painted pixels");
    if(guide.image.channel==2) {
        AdvancedImage rgba;rgba.image=createSolidImage(guide.image.width,guide.image.height,4,0);
        if(!rgba.image.data) throw std::bad_alloc();
        for(size_t i=0;i<pixels;++i) {
            for(int c=0;c<3;++c) rgba.image.data[i*4+c]=guide.image.data[i*2];
            rgba.image.data[i*4+3]=guide.image.data[i*2+1];
        }
        freeImage(guide.image);guide.image=rgba.image;rgba.image.data=nullptr;
    }
    if(encoded_mask.empty()) mask.image=createSolidImage(width,height,1,0);
    else {
        AdvancedImage raw;raw.image=decode_advanced_image(encoded_mask,0);
        if(!raw.image.data) throw std::invalid_argument("Failed to decode Advanced Inpaint mask");
        mask.image=createSolidImage(raw.image.width,raw.image.height,1,0);
        if(!mask.image.data) throw std::bad_alloc();
        for(size_t i=0;i<static_cast<size_t>(raw.image.width)*raw.image.height;++i) {
            const auto* p=raw.image.data+i*raw.image.channel;
            mask.image.data[i]=raw.image.channel<3 ? p[0] : (19595*p[0]+38470*p[1]+7471*p[2]+32768)>>16;
        }
        advanced_resize(mask.image,width,height,false);
    }
    if(!mask.image.data) throw std::bad_alloc();
    advanced_resize(base.image,width,height,true);
    if(guide.image.width!=static_cast<unsigned>(width) || guide.image.height!=static_cast<unsigned>(height)) {
        // Pillow's RGBA resize filters premultiplied colors, not invisible RGB.
        for(size_t i=0;i<pixels;++i) for(int c=0;c<3;++c)
            guide.image.data[i*4+c]=(guide.image.data[i*4+c]*guide.image.data[i*4+3]+127)/255;
        advanced_resize(guide.image,width,height,false);
        for(size_t i=0;i<static_cast<size_t>(width)*height;++i) for(int c=0;c<3;++c) {
            const unsigned a=guide.image.data[i*4+3];
            guide.image.data[i*4+c]=a ? std::min(255u,guide.image.data[i*4+c]*255u/a) : 0;
        }
    }
    for(size_t i=0;i<static_cast<size_t>(width)*height;++i) {
        const unsigned alpha=guide.image.data[i*4+3];
        for(int c=0;c<3;++c) base.image.data[i*3+c]=
            (base.image.data[i*3+c]*(255-alpha)+guide.image.data[i*4+c]*alpha+127)/255;
        if(alpha) mask.image.data[i]=255;
        if(invert) mask.image.data[i]=255-mask.image.data[i];
    }
    // Matches adapter ordering: union first, native inversion/blur second.
    if(!blurMaskImage(mask.image,sigma)) throw std::invalid_argument("Failed to blur Advanced Inpaint mask");
    conditioned=base.image;base.image.data=nullptr;effective_mask=mask.image;mask.image.data=nullptr;
}
