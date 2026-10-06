/* SPDX-License-Identifier: MIT
 * GGML's used subset of the WebGPU C++ API, implemented against the pinned
 * wgpu-native C ABI. This is not Dawn's ABI or a general Dawn replacement. */
#ifndef COSMO_GGML_WEBGPU_CPP_H
#define COSMO_GGML_WEBGPU_CPP_H
#include <webgpu.h>
#include <wgpu.h>
#include "cosmo-webgpu.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace wgpu {
enum class BufferUsage : uint64_t {
    CopyDst = WGPUBufferUsage_CopyDst,
    CopySrc = WGPUBufferUsage_CopySrc,
    MapRead = WGPUBufferUsage_MapRead,
    Storage = WGPUBufferUsage_Storage,
    Uniform = WGPUBufferUsage_Uniform,
};
enum class MapMode : uint64_t {
    Read = WGPUMapMode_Read,
};
enum class CallbackMode : uint32_t {
    AllowSpontaneous = WGPUCallbackMode_AllowSpontaneous,
};
enum class MapAsyncStatus : uint32_t {
    Success = WGPUMapAsyncStatus_Success,
    Error = WGPUMapAsyncStatus_Error,
};
enum class QueueWorkDoneStatus : uint32_t {
    Success = WGPUQueueWorkDoneStatus_Success,
    Error = WGPUQueueWorkDoneStatus_Error,
};
enum class RequestAdapterStatus : uint32_t {
    Success = WGPURequestAdapterStatus_Success,
    Error = WGPURequestAdapterStatus_Error,
};
enum class RequestDeviceStatus : uint32_t {
    Success = WGPURequestDeviceStatus_Success,
    Error = WGPURequestDeviceStatus_Error,
};
enum class DeviceLostReason : uint32_t {
    Destroyed = WGPUDeviceLostReason_Destroyed,
};
enum class ErrorType : uint32_t {
    Validation = WGPUErrorType_Validation,
    Unknown = WGPUErrorType_Unknown,
};
enum class FeatureName : uint32_t {
    ShaderF16 = WGPUFeatureName_ShaderF16,
    Subgroups = WGPUFeatureName_Subgroups,
    TimestampQuery = WGPUFeatureName_TimestampQuery,
};
enum class WaitStatus : uint32_t {
    Success = WGPUWaitStatus_Success,
    TimedOut = WGPUWaitStatus_TimedOut,
    Error = WGPUWaitStatus_Error,
};
inline BufferUsage operator|(BufferUsage a, BufferUsage b) { return BufferUsage(uint64_t(a) | uint64_t(b)); }
struct StringView {
    const char *data = nullptr;
    size_t length = 0;
    StringView() = default;
    StringView(const char *s) : data(s), length(s ? strlen(s) : 0) {}
    StringView(WGPUStringView s) : data(s.data), length(s.length == WGPU_STRLEN ? (s.data ? strlen(s.data) : 0) : s.length) {}
    operator WGPUStringView() const { return {data, length}; }
    operator std::string() const { return std::string(data ? data : "", length); }
};
using Limits = WGPULimits;
using RequestAdapterOptions = WGPURequestAdapterOptions;
using InstanceDescriptor = WGPUInstanceDescriptor;
struct AdapterInfo {
    std::string vendor, architecture, device, description;
    uint32_t vendorID = 0, deviceID = 0, subgroupMinSize = 0, subgroupMaxSize = 0;
    bool software = false;
};
struct FutureState { std::atomic<bool> completed{false}; };
struct Future { std::shared_ptr<FutureState> state; };
template <class F> struct Callback {
    std::shared_ptr<FutureState> state;
    F fn;
};
template <class F> Callback<F> *make_callback(F fn, Future &future) {
    future.state = std::make_shared<FutureState>();
    return new Callback<F>{future.state, std::move(fn)};
}
template <class F, class... Args> void finish_callback(void *userdata, Args &&...args) {
    std::unique_ptr<Callback<F>> callback(static_cast<Callback<F> *>(userdata));
    callback->fn(std::forward<Args>(args)...);
    callback->state->completed.store(true, std::memory_order_release);
}

template <class T, void (*Release)(T)> class Object {
    std::shared_ptr<typename std::remove_pointer<T>::type> object;
public:
    Object() = default;
    Object(std::nullptr_t) {}
    explicit Object(T handle) : object(handle, [](T value) { if (value) Release(value); }) {}
    T Get() const { return object.get(); }
    explicit operator bool() const { return Get() != nullptr; }
    bool operator==(std::nullptr_t) const { return !Get(); }
    bool operator!=(std::nullptr_t) const { return Get() != nullptr; }
    bool operator==(const Object &other) const { return Get() == other.Get(); }
    bool operator!=(const Object &other) const { return Get() != other.Get(); }
};
#define COSMO_WGPU_OBJECT(Name) public Object<WGPU##Name, wgpu##Name##Release> { \
    using Base = Object<WGPU##Name, wgpu##Name##Release>; public: using Base::Base;
class ShaderModule : COSMO_WGPU_OBJECT(ShaderModule) };
class BindGroupLayout : COSMO_WGPU_OBJECT(BindGroupLayout) };
class BindGroup : COSMO_WGPU_OBJECT(BindGroup) };
class CommandBuffer : COSMO_WGPU_OBJECT(CommandBuffer) };
class ComputePipeline : COSMO_WGPU_OBJECT(ComputePipeline)
    std::string label;
    ComputePipeline(WGPUComputePipeline handle, std::string name) : Base(handle), label(std::move(name)) {}
    BindGroupLayout GetBindGroupLayout(uint32_t index) const {
        return BindGroupLayout(wgpuComputePipelineGetBindGroupLayout(Get(), index));
    }
};
class Buffer : COSMO_WGPU_OBJECT(Buffer)
    void Destroy() const { wgpuBufferDestroy(Get()); }
    uint64_t GetSize() const { return wgpuBufferGetSize(Get()); }
    const void *GetConstMappedRange(size_t offset = 0, size_t size = WGPU_WHOLE_MAP_SIZE) const {
        const void *result = wgpuBufferGetConstMappedRange(Get(), offset, size);
        if (result && size) cosmo_webgpu_note_readback();
        return result;
    }
    void Unmap() const { wgpuBufferUnmap(Get()); }
    template <class F> Future MapAsync(MapMode mode, size_t offset, size_t size, CallbackMode, F fn) const {
        Future future;
        auto *callback = make_callback(fn, future);
        WGPUBufferMapCallbackInfo info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        info.mode = WGPUCallbackMode_AllowSpontaneous;
        info.userdata1 = callback;
        info.callback = [](WGPUMapAsyncStatus status, WGPUStringView message, void *u, void *) {
            finish_callback<F>(u, MapAsyncStatus(status), StringView(message));
        };
        wgpuBufferMapAsync(Get(), WGPUMapMode(mode), offset, size, info);
        return future;
    }
};
struct BufferDescriptor {
    StringView label;
    uint64_t size = 0;
    BufferUsage usage{};
    bool mappedAtCreation = false;
};
struct ShaderSourceWGSL {
    StringView code;
};
struct ShaderModuleDescriptor {
    const ShaderSourceWGSL *nextInChain = nullptr;
    StringView label;
};
struct ConstantEntry { StringView key; double value = 0; };
struct ComputePipelineDescriptor {
    StringView label;
    std::nullptr_t layout = nullptr;
    struct {
        ShaderModule module;
        StringView entryPoint;
        size_t constantCount = 0;
        const ConstantEntry *constants = nullptr;
    } compute;
};
struct BindGroupEntry {
    uint32_t binding = 0;
    Buffer buffer;
    uint64_t offset = 0;
    uint64_t size = WGPU_WHOLE_SIZE;
};
struct BindGroupDescriptor {
    StringView label;
    BindGroupLayout layout;
    size_t entryCount = 0;
    const BindGroupEntry *entries = nullptr;
};
class ComputePassEncoder : COSMO_WGPU_OBJECT(ComputePassEncoder)
    mutable bool matmul = false;
    void SetPipeline(const ComputePipeline &pipeline) const {
        wgpuComputePassEncoderSetPipeline(Get(), pipeline.Get());
        matmul = pipeline.label.compare(0, 7, "mul_mat") == 0;
    }
    void SetBindGroup(uint32_t index, const BindGroup &group, size_t count = 0, const uint32_t *offsets = nullptr) const {
        wgpuComputePassEncoderSetBindGroup(Get(), index, group.Get(), count, offsets);
    }
    void DispatchWorkgroups(uint32_t x, uint32_t y = 1, uint32_t z = 1) const {
        wgpuComputePassEncoderDispatchWorkgroups(Get(), x, y, z);
        cosmo_webgpu_note_dispatch();
        if (matmul) cosmo_webgpu_note_matmul_dispatch();
    }
    void End() const { wgpuComputePassEncoderEnd(Get()); }
};
class CommandEncoder : COSMO_WGPU_OBJECT(CommandEncoder)
    ComputePassEncoder BeginComputePass() const {
        WGPUComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        return ComputePassEncoder(wgpuCommandEncoderBeginComputePass(Get(), &desc));
    }
    void CopyBufferToBuffer(const Buffer &src, uint64_t srcOffset, const Buffer &dst, uint64_t dstOffset, uint64_t size) const {
        wgpuCommandEncoderCopyBufferToBuffer(Get(), src.Get(), srcOffset, dst.Get(), dstOffset, size);
    }
    CommandBuffer Finish() const {
        WGPUCommandBufferDescriptor desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        return CommandBuffer(wgpuCommandEncoderFinish(Get(), &desc));
    }
};
class Queue : COSMO_WGPU_OBJECT(Queue)
    void Submit(size_t count, const CommandBuffer *buffers) const {
        std::vector<WGPUCommandBuffer> native;
        native.reserve(count);
        for (size_t i = 0; i < count; ++i) native.push_back(buffers[i].Get());
        wgpuQueueSubmit(Get(), count, native.data());
        if (count) cosmo_webgpu_note_submission();
    }
    void WriteBuffer(const Buffer &buffer, uint64_t offset, const void *data, size_t size) const {
        wgpuQueueWriteBuffer(Get(), buffer.Get(), offset, data, size);
    }
    template <class F> Future OnSubmittedWorkDone(CallbackMode, F fn) const {
        Future future;
        auto *callback = make_callback(fn, future);
        WGPUQueueWorkDoneCallbackInfo info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
        info.mode = WGPUCallbackMode_AllowSpontaneous;
        info.userdata1 = callback;
        info.callback = [](WGPUQueueWorkDoneStatus status, WGPUStringView message, void *u, void *) {
            finish_callback<F>(u, QueueWorkDoneStatus(status), StringView(message));
        };
        wgpuQueueOnSubmittedWorkDone(Get(), info);
        return future;
    }
};
class Device : COSMO_WGPU_OBJECT(Device)
    Queue GetQueue() const { return Queue(wgpuDeviceGetQueue(Get())); }
    Buffer CreateBuffer(const BufferDescriptor *desc) const {
        WGPUBufferDescriptor native = WGPU_BUFFER_DESCRIPTOR_INIT;
        native.label = desc->label;
        native.size = desc->size;
        native.usage = WGPUBufferUsage(desc->usage);
        native.mappedAtCreation = desc->mappedAtCreation;
        return Buffer(wgpuDeviceCreateBuffer(Get(), &native));
    }
    ShaderModule CreateShaderModule(const ShaderModuleDescriptor *desc) const {
        WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
        source.code = desc->nextInChain->code;
        WGPUShaderModuleDescriptor native = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        native.nextInChain = &source.chain;
        native.label = desc->label;
        return ShaderModule(wgpuDeviceCreateShaderModule(Get(), &native));
    }
    ComputePipeline CreateComputePipeline(const ComputePipelineDescriptor *desc) const {
        WGPUComputePipelineDescriptor native = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        native.label = desc->label;
        native.compute.module = desc->compute.module.Get();
        native.compute.entryPoint = desc->compute.entryPoint;
        std::vector<WGPUConstantEntry> constants;
        for (size_t i = 0; i < desc->compute.constantCount; ++i) {
            WGPUConstantEntry value = WGPU_CONSTANT_ENTRY_INIT;
            value.key = desc->compute.constants[i].key;
            value.value = desc->compute.constants[i].value;
            constants.push_back(value);
        }
        native.compute.constantCount = constants.size();
        native.compute.constants = constants.data();
        return ComputePipeline(wgpuDeviceCreateComputePipeline(Get(), &native), std::string(desc->label));
    }
    BindGroup CreateBindGroup(const BindGroupDescriptor *desc) const {
        WGPUBindGroupDescriptor native = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        native.label = desc->label;
        native.layout = desc->layout.Get();
        std::vector<WGPUBindGroupEntry> entries;
        for (size_t i = 0; i < desc->entryCount; ++i) {
            WGPUBindGroupEntry value = WGPU_BIND_GROUP_ENTRY_INIT;
            value.binding = desc->entries[i].binding;
            value.buffer = desc->entries[i].buffer.Get();
            value.offset = desc->entries[i].offset;
            value.size = desc->entries[i].size;
            entries.push_back(value);
        }
        native.entryCount = entries.size();
        native.entries = entries.data();
        return BindGroup(wgpuDeviceCreateBindGroup(Get(), &native));
    }
    CommandEncoder CreateCommandEncoder() const {
        WGPUCommandEncoderDescriptor desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
        return CommandEncoder(wgpuDeviceCreateCommandEncoder(Get(), &desc));
    }
};
struct DeviceDescriptor {
    const Limits *requiredLimits = nullptr;
    const FeatureName *requiredFeatures = nullptr;
    size_t requiredFeatureCount = 0;
    WGPUDeviceLostCallbackInfo lost = WGPU_DEVICE_LOST_CALLBACK_INFO_INIT;
    WGPUUncapturedErrorCallbackInfo error = WGPU_UNCAPTURED_ERROR_CALLBACK_INFO_INIT;
    template <class F> void SetDeviceLostCallback(CallbackMode, F fn) {
        static F callback = fn;
        lost.mode = WGPUCallbackMode_AllowSpontaneous;
        lost.callback = [](WGPUDevice const *device, WGPUDeviceLostReason reason, WGPUStringView message, void *, void *) {
            /* GGML's handler doesn't use the device. Avoid reentrant refcount API calls. */
            (void)device;
            callback(Device(), DeviceLostReason(reason), StringView(message));
        };
    }
    template <class F> void SetUncapturedErrorCallback(F fn) {
        static F callback = fn;
        error.callback = [](WGPUDevice const *device, WGPUErrorType reason, WGPUStringView message, void *, void *) {
            (void)device;
            callback(Device(), ErrorType(reason), StringView(message));
        };
    }
};
class Adapter : COSMO_WGPU_OBJECT(Adapter)
    bool HasFeature(FeatureName feature) const { return wgpuAdapterHasFeature(Get(), WGPUFeatureName(feature)); }
    void GetLimits(Limits *limits) const {
        *limits = WGPU_LIMITS_INIT;
        wgpuAdapterGetLimits(Get(), limits);
    }
    void GetInfo(AdapterInfo *info) const {
        WGPUAdapterInfo native = WGPU_ADAPTER_INFO_INIT;
        wgpuAdapterGetInfo(Get(), &native);
        info->vendor = std::string(StringView(native.vendor));
        info->architecture = std::string(StringView(native.architecture));
        info->device = std::string(StringView(native.device));
        info->description = std::string(StringView(native.description));
        info->vendorID = native.vendorID;
        info->deviceID = native.deviceID;
        info->subgroupMinSize = native.subgroupMinSize;
        info->subgroupMaxSize = native.subgroupMaxSize;
        info->software = native.adapterType == WGPUAdapterType_CPU;
        cosmo_webgpu_note_adapter(info->device.c_str(), info->software);
        wgpuAdapterInfoFreeMembers(native);
    }
    template <class F> Future RequestDevice(const DeviceDescriptor *desc, CallbackMode, F fn) const {
        Future future;
        auto *callback = make_callback(fn, future);
        WGPURequestDeviceCallbackInfo info = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
        info.mode = WGPUCallbackMode_AllowSpontaneous;
        info.userdata1 = callback;
        info.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void *u, void *) {
            finish_callback<F>(u, RequestDeviceStatus(status), Device(device), StringView(message));
        };
        WGPUDeviceDescriptor native = WGPU_DEVICE_DESCRIPTOR_INIT;
        native.requiredLimits = desc->requiredLimits;
        std::vector<WGPUFeatureName> features;
        for (size_t i = 0; i < desc->requiredFeatureCount; ++i) features.push_back(WGPUFeatureName(desc->requiredFeatures[i]));
        native.requiredFeatures = features.data();
        native.requiredFeatureCount = features.size();
        native.deviceLostCallbackInfo = desc->lost;
        native.uncapturedErrorCallbackInfo = desc->error;
        wgpuAdapterRequestDevice(Get(), &native, info);
        return future;
    }
};
class Instance : COSMO_WGPU_OBJECT(Instance)
    template <class F> Future RequestAdapter(const RequestAdapterOptions *options, CallbackMode, F fn) const {
        Future future;
        auto *callback = make_callback(fn, future);
        WGPURequestAdapterCallbackInfo info = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
        info.mode = WGPUCallbackMode_AllowSpontaneous;
        info.userdata1 = callback;
        info.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void *u, void *) {
            std::string text = StringView(message);
            finish_callback<F>(u, RequestAdapterStatus(status), Adapter(adapter), text.c_str());
        };
        WGPURequestAdapterOptions native = options ? *options : WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
        native.backendType = WGPUBackendType_Vulkan;
        native.forceFallbackAdapter = WGPU_TRUE;
        wgpuInstanceRequestAdapter(Get(), &native, info);
        return future;
    }
    WaitStatus WaitAny(const Future &future, uint64_t timeout) const {
        if (!future.state) return WaitStatus::Error;
        const auto start = std::chrono::steady_clock::now();
        while (!future.state->completed.load(std::memory_order_acquire)) {
            /* wgpu-native 29 returns NULL_FUTURE and doesn't implement WaitAny.
             * Its ProcessEvents polls all instance devices and runs callbacks. */
            wgpuInstanceProcessEvents(Get());
            if (future.state->completed.load(std::memory_order_acquire)) return WaitStatus::Success;
            if (timeout != UINT64_MAX && uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count()) >= timeout) return WaitStatus::TimedOut;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return WaitStatus::Success;
    }
};
inline Instance CreateInstance(const InstanceDescriptor *desc) {
    if (cosmo_webgpu_initialize() != 0) return Instance();
    return Instance(wgpuCreateInstance(desc));
}
#undef COSMO_WGPU_OBJECT
}  // namespace wgpu
#endif
