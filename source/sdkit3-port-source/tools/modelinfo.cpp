#include <iostream>
#include <string>
#include <vector>

#include "../include/model_detection.h"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [--print-keys] <model_path>\n";
    std::cout << "\n";
    std::cout << "Description:\n";
    std::cout << "  Inspects a GGUF or safetensors model file and prints information about it.\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  --print-keys  Print all tensor key names (optional)\n";
    std::cout << "  model_path    Path to the model file (.gguf or .safetensors)\n";
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    bool print_keys = false;

    if (argc < 2 || argc > 3) {
        std::cerr << "Error: Invalid number of arguments.\n\n";
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path;
    if (argc == 2) {
        model_path = argv[1];
    } else if (argc == 3) {
        if (std::string(argv[1]) == "--print-keys") {
            print_keys = true;
            model_path = argv[2];
        } else {
            std::cerr << "Error: Invalid argument '" << argv[1] << "'.\n\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Detect model format
    std::string format = detectModelFormat(model_path);

    if (format == "unknown") {
        std::cerr << "Error: Unknown or unsupported file format.\n";
        std::cerr << "Supported formats: GGUF (.gguf), SafeTensors (.safetensors)\n";
        return 1;
    }

    // Extract tensor keys
    std::vector<std::string> tensor_keys = extractTensorKeys(model_path, format);
    if (tensor_keys.empty()) {
        std::cerr << "Error: Failed to read model file or no tensors found.\n";
        return 1;
    }

    // Detect model type
    std::string model_type = inferModelTypeFromTensorKeys(tensor_keys);
    std::cout << "Model type: " << model_type << "\n";

    std::cout << "Number of tensors: " << tensor_keys.size() << "\n";

    if (print_keys) {
        std::cout << "\nTensor keys:\n";
        for (const auto& key : tensor_keys) {
            std::cout << key << "\n";
        }
    }

    return 0;
}
