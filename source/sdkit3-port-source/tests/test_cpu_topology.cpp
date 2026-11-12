#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "core/cpu_topology.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void write_cpu(const std::filesystem::path& root, int cpu, const std::string& mask) {
    auto topology = root / ("cpu" + std::to_string(cpu)) / "topology";
    std::filesystem::create_directories(topology);
    std::ofstream file(topology / "thread_siblings");
    file << mask;
    require(file.good(), "could not write topology fixture");
}

int main() {
    auto root = std::filesystem::temp_directory_path() /
        ("sdkit-cpu-topology-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup {
        std::filesystem::path root;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(root, ignored); }
    } cleanup{root};
    try {
        require(count_sysfs_physical_cores(root.string()) == 0, "missing sysfs must use caller fallback");
        write_cpu(root, 0, "00000003\n");
        write_cpu(root, 1, "00000003\n");
        require(count_sysfs_physical_cores(root.string()) == 1, "SMT siblings counted twice");
        write_cpu(root, 2, "0000000c\n");
        write_cpu(root, 3, "0000000c\n");
        write_cpu(root, 4, "00000010\n");
        write_cpu(root, 5, "00000020\n");
        write_cpu(root, 6, "00000040\n");
        write_cpu(root, 7, "00000080\n");
        require(count_sysfs_physical_cores(root.string()) == 6, "mixed SMT topology fell back to half logical threads");
        write_cpu(root, 8, "");
        require(count_sysfs_physical_cores(root.string()) == 6, "empty topology file counted as a core");
        write_cpu(root, 9, "00000100\n");
        require(count_sysfs_physical_cores(root.string()) == 7, "scan stopped at readable empty topology file");
        write_cpu(root, 11, "00000800\n");
        require(count_sysfs_physical_cores(root.string()) == 7, "existing first-missing-CPU behavior changed");
        std::cout << "PASS Linux sysfs path, SMT/mixed topology, missing/empty files and existing sparse-CPU behavior\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
