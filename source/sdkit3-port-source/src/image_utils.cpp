#include "image_utils.h"

#include <cctype>
#include <limits>
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
