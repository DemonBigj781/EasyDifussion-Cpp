#include "runtime.h"

#include <cstring>

#define STB_IMAGE_STATIC
#define STB_IMAGE_WRITE_STATIC
#define main cosmo_train_main_impl
#include COSMO_TRAINER_SOURCE
#undef main

extern "C" int cosmo_train_main(int argc, char **argv) {
    try {
        if (argc == 2 && (!std::strcmp(argv[1], "--help") || !std::strcmp(argv[1], "-h"))) {
            std::cout <<
                "Native SD 1.5 LoRA training\n"
                "Usage: easy-diffusion.exe train --model CHECKPOINT --dataset DIRECTORY --output DIRECTORY [OPTIONS]\n"
                "  --device CPU                 Backend device (default CPU in this build)\n"
                "  --resolution 256             Training image resolution\n"
                "  --rank 16 --network-alpha 16  LoRA rank and alpha\n"
                "  --steps 1000 --epochs 0       Step limit or epoch schedule\n"
                "  --dataset-repeats 1           Dataset repetition count\n"
                "  --learning-rate 0.0001        UNet learning rate\n"
                "  --text-encoder-learning-rate 0  CLIP rate; zero keeps it frozen\n"
                "  --lr-scheduler constant       constant or cosine_with_restarts\n"
                "  --lr-warmup-steps 0 --lr-scheduler-num-cycles 1\n"
                "  --save-every 100 --save-step-zero 0 --seed 42 --threads 10\n"
                "  --trigger TEXT               Caption trigger\n"
                "This preserves the existing partial trainer; native resume, AdamW8bit,\n"
                "and selectable mixed precision are not implemented.\n";
            return 0;
        }
        // The original standalone trainer defaults to SYCL1. The portable CPU
        // build has no SYCL backend, so select the registered CPU explicitly.
        if ((argc - 1) % 2) {
            std::cerr << "Missing value for native trainer option: " << argv[argc - 1] << '\n';
            return 2;
        }
        const std::set<std::string> supported = {
            "--model", "--dataset", "--output", "--trigger", "--device", "--resolution",
            "--dataset-repeats", "--epochs", "--steps", "--save-every", "--save-step-zero",
            "--rank", "--network-alpha", "--lr-scheduler", "--lr-warmup-steps",
            "--lr-scheduler-num-cycles", "--threads", "--learning-rate",
            "--text-encoder-learning-rate", "--seed",
        };
        std::set<std::string> seen;
        bool selected_device = false;
        for (int i = 1; i < argc; i += 2) {
            if (!supported.count(argv[i])) {
                std::cerr << "Unsupported native trainer option: " << argv[i] << '\n';
                return 2;
            }
            if (!seen.insert(argv[i]).second) {
                std::cerr << "Duplicate native trainer option: " << argv[i] << '\n';
                return 2;
            }
            if (!std::strcmp(argv[i], "--device")) selected_device = true;
        }
        if (selected_device) return cosmo_train_main_impl(argc, argv);
        std::vector<char *> arguments(argv, argv + argc);
        char option[] = "--device";
        char device[] = "CPU";
        arguments.push_back(option);
        arguments.push_back(device);
        arguments.push_back(nullptr);
        return cosmo_train_main_impl(static_cast<int>(arguments.size() - 1), arguments.data());
    } catch (const std::exception &error) {
        std::cerr << "SD 1.5 trainer: " << error.what() << '\n';
        return 1;
    }
}
