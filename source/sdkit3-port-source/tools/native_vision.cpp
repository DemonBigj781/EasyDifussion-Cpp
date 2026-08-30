// Clean C++ Ultralytics-compatible YOLO inference sidecar.
//
// Ultralytics is used only as a one-time exporter for user-supplied .pt
// checkpoints. Runtime inference, image processing, decoding, and NMS below
// do not import or embed the Ultralytics Python package.

#include <torch/script.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

namespace {

struct Options {
    std::string model;
    std::string image;
    float confidence = 0.25f;
    float iou        = 0.45f;
    int input_size   = 640;
    int max_results  = 300;
    int threads      = 0;
};

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int class_id;
};

struct PreparedImage {
    torch::Tensor tensor;
    int original_width;
    int original_height;
    float scale;
    int pad_x;
    int pad_y;
};

const char* const kCoco80[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
};

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    const char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0xf];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

[[noreturn]] void usage(const char* argv0, const std::string& error = {}) {
    if (!error.empty()) std::cerr << error << "\n\n";
    std::cerr << "Usage: " << argv0
              << " --model detector.torchscript --image input.png"
                 " [--conf 0.25] [--iou 0.45] [--size 640] [--max 300] [--threads N]\n";
    std::exit(error.empty() ? 0 : 2);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) usage(argv[0], std::string("Missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--model") options.model = value("--model");
        else if (arg == "--image") options.image = value("--image");
        else if (arg == "--conf") options.confidence = std::stof(value("--conf"));
        else if (arg == "--iou") options.iou = std::stof(value("--iou"));
        else if (arg == "--size") options.input_size = std::stoi(value("--size"));
        else if (arg == "--max") options.max_results = std::stoi(value("--max"));
        else if (arg == "--threads") options.threads = std::stoi(value("--threads"));
        else if (arg == "--help" || arg == "-h") usage(argv[0]);
        else usage(argv[0], "Unknown argument: " + arg);
    }
    if (options.model.empty() || options.image.empty()) usage(argv[0], "--model and --image are required");
    if (!(options.confidence >= 0.f && options.confidence <= 1.f)) usage(argv[0], "--conf must be in [0,1]");
    if (!(options.iou >= 0.f && options.iou <= 1.f)) usage(argv[0], "--iou must be in [0,1]");
    if (options.input_size < 32 || options.input_size > 4096 || options.input_size % 32 != 0)
        usage(argv[0], "--size must be a multiple of 32 in [32,4096]");
    if (options.max_results < 1 || options.max_results > 10000) usage(argv[0], "--max must be in [1,10000]");
    return options;
}

PreparedImage prepare_image(const Options& options) {
    int width = 0, height = 0, channels = 0;
    unsigned char* decoded = stbi_load(options.image.c_str(), &width, &height, &channels, 3);
    if (!decoded || width <= 0 || height <= 0) {
        stbi_image_free(decoded);
        throw std::runtime_error("Could not decode input image: " + options.image);
    }

    const float scale = std::min(static_cast<float>(options.input_size) / width,
                                 static_cast<float>(options.input_size) / height);
    const int resized_width  = std::max(1, static_cast<int>(std::round(width * scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(height * scale)));
    const int pad_x          = (options.input_size - resized_width) / 2;
    const int pad_y          = (options.input_size - resized_height) / 2;

    std::vector<unsigned char> resized(static_cast<size_t>(resized_width) * resized_height * 3);
    if (!stbir_resize_uint8(decoded, width, height, 0, resized.data(), resized_width, resized_height, 0, 3)) {
        stbi_image_free(decoded);
        throw std::runtime_error("Image resize failed");
    }
    stbi_image_free(decoded);

    std::vector<float> nchw(static_cast<size_t>(3) * options.input_size * options.input_size,
                            114.f / 255.f);
    const size_t plane = static_cast<size_t>(options.input_size) * options.input_size;
    for (int y = 0; y < resized_height; ++y) {
        for (int x = 0; x < resized_width; ++x) {
            const size_t src = (static_cast<size_t>(y) * resized_width + x) * 3;
            const size_t dst = static_cast<size_t>(y + pad_y) * options.input_size + (x + pad_x);
            nchw[dst]             = resized[src] / 255.f;
            nchw[plane + dst]     = resized[src + 1] / 255.f;
            nchw[2 * plane + dst] = resized[src + 2] / 255.f;
        }
    }

    auto tensor = torch::from_blob(nchw.data(), {1, 3, options.input_size, options.input_size},
                                   torch::TensorOptions().dtype(torch::kFloat32)).clone();
    return {std::move(tensor), width, height, scale, pad_x, pad_y};
}

torch::Tensor extract_prediction(const torch::jit::IValue& value) {
    if (value.isTensor()) return value.toTensor();
    if (value.isTuple()) {
        for (const auto& element : value.toTuple()->elements()) {
            if (element.isTensor()) return element.toTensor();
        }
    }
    if (value.isList()) {
        for (const auto& element : value.toListRef()) {
            if (element.isTensor()) return element.toTensor();
        }
    }
    throw std::runtime_error("TorchScript output does not contain a prediction tensor");
}

float intersection_over_union(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
    const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return intersection / std::max(area_a + area_b - intersection, 1e-9f);
}

std::vector<Detection> decode_and_nms(torch::Tensor prediction,
                                      const PreparedImage& image,
                                      const Options& options,
                                      int* class_count_out) {
    prediction = prediction.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    while (prediction.dim() > 2 && prediction.size(0) == 1) prediction = prediction.squeeze(0);
    if (prediction.dim() != 2) throw std::runtime_error("Expected a 2D YOLO prediction tensor");

    // Ultralytics detect exports are [4 + classes, anchors]. Accept the
    // transposed layout as well for other compatible exporters.
    if (prediction.size(0) < prediction.size(1) && prediction.size(0) <= 1024) {
        prediction = prediction.transpose(0, 1).contiguous();
    }
    const int64_t anchors  = prediction.size(0);
    const int64_t features = prediction.size(1);
    if (features < 5) throw std::runtime_error("Prediction has fewer than five features per anchor");
    const int classes = static_cast<int>(features - 4);
    *class_count_out = classes;

    auto data = prediction.accessor<float, 2>();
    std::vector<Detection> candidates;
    candidates.reserve(static_cast<size_t>(std::min<int64_t>(anchors, 4096)));
    for (int64_t i = 0; i < anchors; ++i) {
        int best_class = 0;
        float best_score = data[i][4];
        for (int c = 1; c < classes; ++c) {
            if (data[i][4 + c] > best_score) {
                best_score = data[i][4 + c];
                best_class = c;
            }
        }
        if (!std::isfinite(best_score) || best_score < options.confidence) continue;

        const float cx = data[i][0];
        const float cy = data[i][1];
        const float w  = data[i][2];
        const float h  = data[i][3];
        Detection detection{
            (cx - w * 0.5f - image.pad_x) / image.scale,
            (cy - h * 0.5f - image.pad_y) / image.scale,
            (cx + w * 0.5f - image.pad_x) / image.scale,
            (cy + h * 0.5f - image.pad_y) / image.scale,
            best_score,
            best_class,
        };
        detection.x1 = std::clamp(detection.x1, 0.f, static_cast<float>(image.original_width));
        detection.y1 = std::clamp(detection.y1, 0.f, static_cast<float>(image.original_height));
        detection.x2 = std::clamp(detection.x2, 0.f, static_cast<float>(image.original_width));
        detection.y2 = std::clamp(detection.y2, 0.f, static_cast<float>(image.original_height));
        if (detection.x2 > detection.x1 && detection.y2 > detection.y1) candidates.push_back(detection);
    }

    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    std::vector<Detection> kept;
    kept.reserve(static_cast<size_t>(std::min<int>(options.max_results, candidates.size())));
    for (const auto& candidate : candidates) {
        bool suppressed = false;
        for (const auto& accepted : kept) {
            if (candidate.class_id == accepted.class_id &&
                intersection_over_union(candidate, accepted) > options.iou) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (static_cast<int>(kept.size()) >= options.max_results) break;
        }
    }
    return kept;
}

std::string class_label(int class_id, int class_count) {
    if (class_count == 1) return "face";
    if (class_count == 80 && class_id >= 0 && class_id < 80) return kCoco80[class_id];
    return "class_" + std::to_string(class_id);
}

void print_json(const Options& options,
                const PreparedImage& image,
                const std::vector<Detection>& detections,
                int class_count,
                double inference_ms) {
    std::cout << std::fixed << std::setprecision(6)
              << "{\"model\":\"" << json_escape(options.model) << "\","
              << "\"width\":" << image.original_width << ",\"height\":" << image.original_height << ','
              << "\"input_size\":" << options.input_size << ",\"classes\":" << class_count << ','
              << "\"inference_ms\":" << inference_ms << ",\"detections\":[";
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        if (i) std::cout << ',';
        std::cout << "{\"x\":" << d.x1 << ",\"y\":" << d.y1
                  << ",\"width\":" << (d.x2 - d.x1) << ",\"height\":" << (d.y2 - d.y1)
                  << ",\"confidence\":" << d.confidence << ",\"class_id\":" << d.class_id
                  << ",\"label\":\"" << json_escape(class_label(d.class_id, class_count)) << "\"}";
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const unsigned hardware_threads = std::max(1u, std::thread::hardware_concurrency());
        at::set_num_threads(options.threads > 0 ? options.threads
                                                : static_cast<int>(std::min(4u, hardware_threads)));
        at::set_num_interop_threads(1);

        PreparedImage image = prepare_image(options);
        torch::jit::script::Module module = torch::jit::load(options.model, torch::kCPU);
        module.eval();

        torch::InferenceMode guard;
        const auto start = std::chrono::steady_clock::now();
        const auto output = module.forward({image.tensor});
        torch::Tensor prediction = extract_prediction(output);
        const auto end = std::chrono::steady_clock::now();
        const double inference_ms = std::chrono::duration<double, std::milli>(end - start).count();

        int class_count = 0;
        const auto detections = decode_and_nms(std::move(prediction), image, options, &class_count);
        print_json(options, image, detections, class_count, inference_ms);
        return 0;
    } catch (const c10::Error& error) {
        std::cerr << "LibTorch error: " << error.what_without_backtrace() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Native vision error: " << error.what() << '\n';
    }
    return 1;
}
