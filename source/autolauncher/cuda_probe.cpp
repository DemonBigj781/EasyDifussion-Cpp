// Native worker: dynamically query the installed driver, not the CUDA SDK.
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#define CUDA_CALL __stdcall
#else
#include <dlfcn.h>
#define CUDA_CALL
#endif

int main(int argc, char** argv) {
    try {
        int minimum_driver = 0, ordinal = 0;
        std::vector<int> architectures;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--help") {
                std::cout << "cuda-probe --sm 86 [--sm 89 ...] --min-driver 12040 [--device 0]\n";
                return 0;
            }
            if (arg != "--sm" && arg != "--min-driver" && arg != "--device")
                throw std::runtime_error("Unknown option: " + arg);
            if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
            std::string value(argv[i]);
            if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                throw std::runtime_error("Expected nonnegative integer");
            int n = std::stoi(value);
            if (arg == "--sm") architectures.push_back(n);
            else if (arg == "--min-driver") minimum_driver = n;
            else ordinal = n;
        }
        if (architectures.empty() || minimum_driver <= 0)
            throw std::runtime_error("Declare at least one exact --sm and a --min-driver requirement");
#ifdef _WIN32
        auto library = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        auto symbol = [&](const char* name) { return GetProcAddress(library, name); };
#else
        auto library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        auto symbol = [&](const char* name) { return dlsym(library, name); };
#endif
        if (!library) { std::cerr << "CUDA driver library unavailable\n"; return 1; }
        auto init = reinterpret_cast<int(CUDA_CALL *)(unsigned)>(symbol("cuInit"));
        auto version = reinterpret_cast<int(CUDA_CALL *)(int*)>(symbol("cuDriverGetVersion"));
        auto device = reinterpret_cast<int(CUDA_CALL *)(int*, int)>(symbol("cuDeviceGet"));
        auto attribute = reinterpret_cast<int(CUDA_CALL *)(int*, int, int)>(symbol("cuDeviceGetAttribute"));
        int driver = 0, gpu = 0, major = 0, minor = 0;
        if (!init || !version || !device || !attribute || init(0) || version(&driver) ||
            driver < minimum_driver || device(&gpu, ordinal) ||
            attribute(&major, 75, gpu) || attribute(&minor, 76, gpu)) {
            std::cerr << "CUDA device/driver requirements not met\n";
            return 1;
        }
        int sm = major * 10 + minor;
        if (std::find(architectures.begin(), architectures.end(), sm) == architectures.end()) {
            std::cerr << "Visible CUDA device " << ordinal << " has unsupported sm" << sm << '\n';
            return 1;
        }
        std::cout << "{\"device\":" << ordinal << ",\"sm\":" << sm << ",\"driver\":" << driver << "}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "cuda-probe: " << e.what() << '\n';
        return 2;
    }
}
