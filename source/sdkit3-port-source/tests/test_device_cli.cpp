#include "device_cli.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

int main() {
    std::string backend, spec;
    require(sdkit::parse_driver_name("Vulkan", false, backend).empty() && backend == "vulkan",
            "Vulkan spelling did not map to its native backend");
    require(sdkit::parse_driver_name("oneAPI", false, backend).empty() && backend == "sycl",
            "oneAPI did not map to SYCL");
    require(sdkit::parse_driver_name("OpenVINO", false, backend).find("not wired") != std::string::npos,
            "OpenVINO was not rejected explicitly");
    require(sdkit::parse_driver_name("sycl", false, backend).find("oneapi") != std::string::npos,
            "retired public SYCL spelling did not point to oneapi");
    const std::vector<sdkit::PhysicalDevice> inventory{
        {"cuda", "CUDA0", "0000:01:00.0"}, {"cuda", "CUDA1", "0000:0a:00.0"},
        {"vulkan", "Vulkan0", "0000:0a:00.0"}, {"sycl", "SYCL0", ""},
    };
    const std::string request = "0000:0A:00.0, 00000000:01:00.0";
    require(sdkit::resolve_pcie_devices("cuda", request, inventory, spec).empty() && spec == "CUDA1&CUDA0",
            "ordered addresses did not bind to physical cards");
    const std::vector<sdkit::PhysicalDevice> renumbered{
        {"cuda", "CUDA2", "00000000:0a:00.0"}, {"vulkan", "Vulkan2", "0000:0a:00.0"},
        {"cuda", "CUDA4", "0000:01:00.0"}
    };
    require(sdkit::resolve_pcie_devices("cuda", request, renumbered, spec).empty() && spec == "CUDA2&CUDA4",
            "renumbering/reordering the registry changed the desired physical pairing");
    require(sdkit::resolve_pcie_devices("vulkan", "0000:0A:00.0", inventory, spec).empty() &&
            spec == "Vulkan0", "address was routed to another driver");
    require(!sdkit::resolve_pcie_devices("cuda", "0", inventory, spec).empty() &&
            !sdkit::resolve_pcie_devices("cuda", "0000:01:20.0", inventory, spec).empty() &&
            !sdkit::resolve_pcie_devices("cuda", "0000:01:00.8", inventory, spec).empty(),
            "numeric or malformed input was accepted");
    require(sdkit::resolve_pcie_devices("cuda", "0000:0a:00.0,0000:0A:00.0", inventory, spec).find("Duplicate")
                != std::string::npos, "physical address was duplicated through alternate casing");
    require(sdkit::resolve_pcie_devices("sycl", "0000:01:00.0", inventory, spec).find("no physical")
                != std::string::npos, "invented missing PCIe metadata");
    require(sdkit::resolve_pcie_devices("cuda", "0000:02:00.0", inventory, spec).find("not reported")
                != std::string::npos, "missing address silently selected another device");
    require(!sdkit::resolve_pcie_devices("auto", "0000:01:00.0", inventory, spec).empty(),
            "auto driver silently chose an accelerator");
    auto virtualized = inventory;
    virtualized.push_back({"cuda", "CUDA2", "00000000:01:00.0-v2"});
    require(sdkit::resolve_pcie_devices("cuda", "0000:01:00.0", virtualized, spec).find("multiple")
                != std::string::npos && sdkit::selectable_pcie_address(virtualized[0], virtualized).empty(),
            "ambiguous virtual PCIe address was advertised or accepted");
    virtualized.erase(virtualized.begin());
    require(sdkit::resolve_pcie_devices("cuda", "0000:01:00.0", virtualized, spec).find("virtual")
                != std::string::npos, "a single virtual slice was mistaken for the physical device");
    require(sdkit::physical_pcie_address("00010abc:05:00.2") == "10abc:05:00.2",
            "a genuine nonzero upper domain was lost");
    require(sdkit::selectable_pcie_address(inventory[0], inventory) == "0000:01:00.0",
            "list did not report a selectable physical address");
    std::puts("PASS: physical PCIe routing survives renumbering and rejects missing or ambiguous identities");
}
