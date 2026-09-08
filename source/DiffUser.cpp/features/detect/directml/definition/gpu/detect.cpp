#include "detect.hpp"

#ifndef EDCPP_DETECT_DIRECTML
#define EDCPP_DETECT_DIRECTML 0
#endif
#if EDCPP_DETECT_DIRECTML
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

namespace edcpp::api::detect::directml::definition::gpu {

#if EDCPP_DETECT_DIRECTML
namespace {
std::string utf8(const wchar_t* value) {
    if (value == nullptr || *value == L'\0') return {};
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
    if (required <= 1) return {};
    std::string converted(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value, -1, converted.data(), required,
                        nullptr, nullptr);
    while (!converted.empty() && converted.back() == '\0') converted.pop_back();
    return converted;
}
} // namespace
#endif

NativeResult detect() {
    NativeResult result;
#if EDCPP_DETECT_DIRECTML
    using Microsoft::WRL::ComPtr;

    ComPtr<IDXGIFactory6> factory;
    const HRESULT factory_rc = CreateDXGIFactory2(
        0, __uuidof(IDXGIFactory6),
        reinterpret_cast<void**>(factory.GetAddressOf()));
    if (FAILED(factory_rc)) {
        result.diagnostic = "CreateDXGIFactory2 failed with HRESULT " +
                            std::to_string(static_cast<unsigned long>(factory_rc));
        return result;
    }

    for (UINT ordinal = 0;; ++ordinal) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT adapter_rc = factory->EnumAdapterByGpuPreference(
            ordinal, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            __uuidof(IDXGIAdapter1),
            reinterpret_cast<void**>(adapter.GetAddressOf()));
        if (adapter_rc == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(adapter_rc)) continue;

        DXGI_ADAPTER_DESC1 description{};
        if (FAILED(adapter->GetDesc1(&description)) ||
            (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }

        ComPtr<ID3D12Device> d3d12_device;
        if (FAILED(D3D12CreateDevice(
                adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                __uuidof(ID3D12Device),
                reinterpret_cast<void**>(d3d12_device.GetAddressOf())))) {
            continue;
        }

        ComPtr<IDMLDevice> directml_device;
        if (FAILED(DMLCreateDevice(
                d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                __uuidof(IDMLDevice),
                reinterpret_cast<void**>(directml_device.GetAddressOf())))) {
            continue;
        }

        NativeDevice device;
        device.index = static_cast<int>(result.devices.size());
        device.name = utf8(description.Description);
        std::ostringstream identity;
        identity << "vendor_" << std::hex << std::setw(4) << std::setfill('0')
                 << description.VendorId << "/device_" << std::setw(4)
                 << description.DeviceId;
        device.architecture = identity.str();
        device.total_memory = static_cast<std::uint64_t>(description.DedicatedVideoMemory);
        device.available = true;
        result.devices.push_back(std::move(device));
    }

    result.available = !result.devices.empty();
    if (!result.available) {
        result.diagnostic = "DirectML reported no compatible hardware adapters";
    }
#else
    result.diagnostic = "DirectML detection was not enabled for this compiler target";
#endif
    return result;
}

} // namespace edcpp::api::detect::directml::definition::gpu
