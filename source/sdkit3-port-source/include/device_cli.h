#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sdkit {
inline std::string driver_lowercase(std::string_view name) {
    std::string normalized(name);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized;
}

inline std::string native_backend_for_driver(std::string_view driver) {
    if (driver == "oneapi") return "sycl";
    if (driver == "auto" || driver == "cpu" || driver == "cuda" || driver == "vulkan") {
        return std::string(driver);
    }
    return {};
}

inline std::string parse_driver_name(std::string_view name, bool filter, std::string& backend) {
    const auto driver = driver_lowercase(name);
    if (driver == "openvino") {
        return "OpenVINO is not wired into the native EasyDiffusion runtime";
    }
    if (driver == "sycl") {
        return "Use the public driver name 'oneapi' for the native SYCL backend";
    }
    auto selected = native_backend_for_driver(driver);
    if (selected.empty() || (filter && selected == "auto")) {
        return "Invalid driver '" + std::string(name) + "' (expected " +
               (filter ? "" : "auto, ") + "cpu, cuda, vulkan, oneapi, or openvino)";
    }
    backend = std::move(selected);
    return {};
}

inline std::string_view retired_device_option_replacement(std::string_view option) {
    if (option == "--backend") return "--driver <name> (use oneapi for SYCL)";
    if (option == "--list-devices" || option == "--backend-info") return "--devices list [driver]";
    return {};
}

inline std::string_view trim_address(std::string_view address) {
    while (!address.empty() && std::isspace(static_cast<unsigned char>(address.front()))) address.remove_prefix(1);
    while (!address.empty() && std::isspace(static_cast<unsigned char>(address.back()))) address.remove_suffix(1);
    return address;
}

inline std::string physical_pcie_address(std::string_view address) {
    const auto domain = address.find(':');
    if (domain < 4 || domain > 8 || address.size() != domain + 8 ||
        address[domain + 3] != ':' || address[domain + 6] != '.' ||
        address[domain + 7] < '0' || address[domain + 7] > '7' ||
        address[domain + 4] < '0' || address[domain + 4] > '1') {
        return {};
    }
    auto value = driver_lowercase(address);
    for (size_t i = 0; i < value.size(); ++i) {
        if (i == domain || i == domain + 3 || i == domain + 6) continue;
        if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f'))) {
            return {};
        }
    }
    // Drivers may print a 32-bit domain padded to eight hex digits. Strip
    // only redundant zeros while retaining the full nonzero domain.
    size_t trim = 0;
    while (domain - trim > 4 && value[trim] == '0') ++trim;
    value.erase(0, trim);
    return value;
}

struct PhysicalDevice {
    std::string driver;
    std::string selector;
    std::string metadata_id;
};

inline bool device_matches_address(const PhysicalDevice& device, std::string_view address) {
    const auto metadata = driver_lowercase(device.metadata_id);
    // Virtual CUDA devices expose PCI IDs suffixed with -v<index>. Include
    // them in the match count, but never route a physical request to one.
    const auto suffix = metadata.find("-v");
    if (suffix == std::string::npos) return physical_pcie_address(metadata) == address;
    return suffix + 2 < metadata.size() &&
           std::all_of(metadata.begin() + suffix + 2, metadata.end(),
                       [](char ch) { return ch >= '0' && ch <= '9'; }) &&
           physical_pcie_address(std::string_view(metadata).substr(0, suffix)) == address;
}

inline std::string selectable_pcie_address(const PhysicalDevice& device,
                                           const std::vector<PhysicalDevice>& inventory) {
    const auto address = physical_pcie_address(device.metadata_id);
    if (address.empty()) return {};
    const auto matches = std::count_if(inventory.begin(), inventory.end(), [&](const auto& item) {
        return item.driver == device.driver && device_matches_address(item, address);
    });
    return matches == 1 ? address : std::string{};
}

inline std::string resolve_pcie_devices(const std::string& driver, std::string_view requested,
                                        const std::vector<PhysicalDevice>& inventory, std::string& spec) {
    if (driver == "cpu" || driver == "auto") {
        return "--device requires an explicit accelerator --driver; CPU and auto do not select PCIe devices";
    }
    std::vector<std::string> seen;
    std::string resolved;
    for (;;) {
        const auto separator = requested.find(',');
        const auto address = physical_pcie_address(trim_address(requested.substr(0, separator)));
        if (address.empty()) {
            return "--device requires a full physical PCIe address (e.g. 0000:01:00.0), not an ordinal or virtual ID";
        }
        if (std::find(seen.begin(), seen.end(), address) != seen.end()) {
            return "Duplicate PCIe address '" + address + "'";
        }
        seen.push_back(address);
        const PhysicalDevice* match = nullptr;
        bool exposes_physical_ids = false;
        for (const auto& device : inventory) {
            if (device.driver != driver) continue;
            exposes_physical_ids |= !physical_pcie_address(device.metadata_id).empty();
            if (!device_matches_address(device, address)) continue;
            if (match) return "PCIe address '" + address + "' matches multiple devices for this driver";
            match = &device;
        }
        if (!match) {
            if (!exposes_physical_ids) return "The selected driver exposes no physical PCIe addresses in this build";
            return "PCIe address '" + address + "' was not reported by the selected driver";
        }
        if (physical_pcie_address(match->metadata_id).empty()) {
            return "PCIe address '" + address + "' is exposed only as a virtual device; refusing to guess";
        }
        if (match->selector.empty() || match->selector.find_first_of("&,") != std::string::npos ||
            std::count_if(inventory.begin(), inventory.end(), [&](const auto& device) {
                return device.driver == driver && device.selector == match->selector;
            }) != 1) {
            return "The selected PCIe device has no unambiguous native selector";
        }
        if (!resolved.empty()) resolved += '&';
        resolved += match->selector;
        if (separator == std::string_view::npos) break;
        requested.remove_prefix(separator + 1);
    }
    spec = std::move(resolved);
    return {};
}
}  // namespace sdkit
