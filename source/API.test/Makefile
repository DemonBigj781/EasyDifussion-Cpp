CXX ?= c++
NVCC ?= nvcc
CMAKE ?= cmake
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
CUDA_CPPFLAGS ?= -DEDCPP_DETECT_CUDA=1 -DEDCPP_LOAD_CUDA=1 -DEDCPP_UNLOAD_CUDA=1
CUDA_LDLIBS ?= -lcudart
OPENCL_LDLIBS ?= -lOpenCL
VULKAN_LDLIBS ?= -lvulkan
OPENGL_LDLIBS ?= -lEGL -lGL
MESA_LDLIBS ?= -lEGL -lGL

API_ROOT := ../API.cpp
BUILD_DIR ?= build
INCLUDES := -I. -I$(API_ROOT) -I$(API_ROOT)/library/include
COMMON_SOURCES := \
	$(API_ROOT)/common/api.cpp \
	$(API_ROOT)/features/detect/common/detect.cpp
CPU_SOURCES := \
	Detect/cpu/detect.cpp \
	$(COMMON_SOURCES) \
	$(API_ROOT)/features/detect/cpu/definition/cpu/detect.cpp \
	$(API_ROOT)/features/detect/cpu/translation/cpu/detect.cpp \
	$(API_ROOT)/library/src/cpu_handler.cpp
CUDA_SOURCES := \
	Detect/cuda/detect.cpp \
	$(COMMON_SOURCES) \
	$(API_ROOT)/features/detect/cuda/definition/gpu/detect.cpp \
	$(API_ROOT)/features/detect/cuda/translation/gpu/detect.cpp \
	$(API_ROOT)/library/src/cuda_handler.cpp
OPENCL_SOURCES := \
	Detect/opencl/detect.cpp \
	$(COMMON_SOURCES) \
	$(API_ROOT)/features/detect/opencl/definition/cpu/detect.cpp \
	$(API_ROOT)/features/detect/opencl/definition/gpu/detect.cpp \
	$(API_ROOT)/features/detect/opencl/definition/npu/detect.cpp \
	$(API_ROOT)/features/detect/opencl/translation/cpu/detect.cpp \
	$(API_ROOT)/features/detect/opencl/translation/gpu/detect.cpp \
	$(API_ROOT)/features/detect/opencl/translation/npu/detect.cpp \
	$(API_ROOT)/features/detect/opencl/translation/register.cpp \
	$(API_ROOT)/library/src/opencl_handler.cpp
VULKAN_SOURCES := \
	Detect/vulkan/detect.cpp \
	$(COMMON_SOURCES) \
	$(API_ROOT)/features/detect/vulkan/definition/cpu/detect.cpp \
	$(API_ROOT)/features/detect/vulkan/definition/gpu/detect.cpp \
	$(API_ROOT)/features/detect/vulkan/translation/cpu/detect.cpp \
	$(API_ROOT)/features/detect/vulkan/translation/gpu/detect.cpp \
	$(API_ROOT)/features/detect/vulkan/translation/register.cpp \
	$(API_ROOT)/library/src/vulkan_handler.cpp
OPENGL_SOURCES := \
	Detect/opengl/detect.cpp \
	$(COMMON_SOURCES) \
	$(API_ROOT)/features/detect/opengl/definition/gpu/detect.cpp \
	$(API_ROOT)/features/detect/opengl/translation/gpu/detect.cpp \
	$(API_ROOT)/library/src/opengl_handler.cpp
MESA_SOURCES := \
	Detect/mesa/detect.cpp \
	$(COMMON_SOURCES) \
	$(API_ROOT)/features/detect/mesa/definition/cpu/detect.cpp \
	$(API_ROOT)/features/detect/mesa/definition/gpu/detect.cpp \
	$(API_ROOT)/features/detect/mesa/translation/cpu/detect.cpp \
	$(API_ROOT)/features/detect/mesa/translation/gpu/detect.cpp \
	$(API_ROOT)/features/detect/mesa/translation/register.cpp \
	$(API_ROOT)/library/src/mesa_handler.cpp
MODEL_COMMON_SOURCES := \
	$(API_ROOT)/features/load/common/load.cpp \
	$(API_ROOT)/features/unload/common/unload.cpp
CPU_MODEL_SOURCES := \
	$(MODEL_COMMON_SOURCES) \
	$(API_ROOT)/features/load/model/cpu/definition/cpu/load.cpp \
	$(API_ROOT)/features/load/model/cpu/translation/cpu/load.cpp \
	$(API_ROOT)/features/unload/model/cpu/definition/cpu/unload.cpp \
	$(API_ROOT)/features/unload/model/cpu/translation/cpu/unload.cpp \
	$(API_ROOT)/library/src/cpu_model_lifecycle.cpp
CUDA_MODEL_SOURCES := \
	$(MODEL_COMMON_SOURCES) \
	$(API_ROOT)/features/load/model/cuda/definition/gpu/load.cpp \
	$(API_ROOT)/features/load/model/cuda/translation/gpu/load.cpp \
	$(API_ROOT)/features/unload/model/cuda/definition/gpu/unload.cpp \
	$(API_ROOT)/features/unload/model/cuda/translation/gpu/unload.cpp \
	$(API_ROOT)/library/src/cuda_model_lifecycle.cpp
LOAD_CPU_SOURCES := Load/Cpu/load.cpp $(CPU_MODEL_SOURCES)
LOAD_CUDA_SOURCES := Load/Cuda/load.cpp $(CUDA_MODEL_SOURCES)
UNLOAD_CPU_SOURCES := Unload/Cpu/unload.cpp $(CPU_MODEL_SOURCES)
UNLOAD_CUDA_SOURCES := Unload/Cuda/unload.cpp $(CUDA_MODEL_SOURCES)
OVERFLOW_COMMON_SOURCES := \
	$(API_ROOT)/features/overflow/common/plan.cpp \
	$(API_ROOT)/features/overflow/common/normalize.cpp \
	$(API_ROOT)/features/overflow/common/dispatch.cpp \
	$(API_ROOT)/library/src/overflow.cpp
CPU_OVERFLOW_SOURCES := \
	Overflow/Cpu/overflow.cpp \
	$(OVERFLOW_COMMON_SOURCES) \
	$(API_ROOT)/features/overflow/cpu/definition/cpu/allocate.cpp \
	$(API_ROOT)/features/overflow/cpu/definition/cpu/release.cpp \
	$(API_ROOT)/features/overflow/cpu/translation/cpu/allocate.cpp \
	$(API_ROOT)/features/overflow/cpu/translation/cpu/release.cpp \
	$(API_ROOT)/library/src/cpu_overflow.cpp
CUDA_OVERFLOW_SOURCES := \
	Overflow/Cuda/overflow.cu \
	$(OVERFLOW_COMMON_SOURCES) \
	$(API_ROOT)/features/overflow/cuda/definition/gpu/allocate.cpp \
	$(API_ROOT)/features/overflow/cuda/definition/gpu/release.cpp \
	$(API_ROOT)/features/overflow/cuda/translation/gpu/allocate.cpp \
	$(API_ROOT)/features/overflow/cuda/translation/gpu/release.cpp \
	$(API_ROOT)/library/src/cuda_overflow.cpp
XFORMERS_COMMON_SOURCES := \
	$(API_ROOT)/features/attention/xformers/common/qkt.cpp \
	$(API_ROOT)/features/attention/xformers/common/mask.cpp \
	$(API_ROOT)/features/attention/xformers/common/softmax.cpp \
	$(API_ROOT)/features/attention/xformers/common/av.cpp \
	$(API_ROOT)/features/attention/xformers/common/forward.cpp
XFORMERS_CPU_RUNTIME_SOURCES := \
	$(XFORMERS_COMMON_SOURCES) \
	$(API_ROOT)/features/attention/xformers/cpu/translation/cpu/qkt.cpp \
	$(API_ROOT)/features/attention/xformers/cpu/translation/cpu/mask.cpp \
	$(API_ROOT)/features/attention/xformers/cpu/translation/cpu/softmax.cpp \
	$(API_ROOT)/features/attention/xformers/cpu/translation/cpu/av.cpp \
	$(API_ROOT)/features/attention/xformers/cpu/translation/cpu/forward.cpp
XFORMERS_CPU_CYCLE_SOURCES := \
	MultiTest/xformers/cycle/xformers-load-unload-cpu-test.cpp \
	$(XFORMERS_CPU_RUNTIME_SOURCES) \
	$(CPU_MODEL_SOURCES)
XFORMERS_IMAGE_DIR := MultiTest/xformers/generate-image
XFORMERS_IMAGE_SOURCES := \
	$(XFORMERS_IMAGE_DIR)/CMakeLists.txt \
	$(XFORMERS_IMAGE_DIR)/xformers-image-gen.hpp \
	$(XFORMERS_IMAGE_DIR)/xformers-image-gen.cpp \
	$(XFORMERS_IMAGE_DIR)/xformers-cpu-gen.cpp \
	$(XFORMERS_IMAGE_DIR)/xformers-cuda-gen.cpp
XFORMERS_CUDA_DEFINITION_SOURCES := \
	$(API_ROOT)/features/attention/xformers/cuda/definition/gpu/qkt.cpp \
	$(API_ROOT)/features/attention/xformers/cuda/definition/gpu/mask.cpp \
	$(API_ROOT)/features/attention/xformers/cuda/definition/gpu/softmax.cpp \
	$(API_ROOT)/features/attention/xformers/cuda/definition/gpu/av.cpp \
	$(API_ROOT)/features/attention/xformers/cuda/definition/gpu/forward.cpp
XFORMERS_CUDA_RUNTIME_SOURCES := \
	$(XFORMERS_COMMON_SOURCES) \
	$(XFORMERS_CUDA_DEFINITION_SOURCES) \
	$(API_ROOT)/features/attention/xformers/cuda/translation/gpu/forward.cu
XFORMERS_CUDA_CYCLE_SOURCES := \
	MultiTest/xformers/cycle/xformers-load-unload-cuda-test.cpp \
	$(XFORMERS_CUDA_RUNTIME_SOURCES) \
	$(CUDA_MODEL_SOURCES)

.PHONY: all cpu cuda opencl vulkan opengl mesa \
	load-cpu load-cuda unload-cpu unload-cuda overflow-cpu overflow-cuda \
	xformers-cpu xformers-cuda xformers-cpu-gen xformers-cuda-gen \
	run-cpu run-cuda run-opencl run-vulkan run-opengl run-mesa \
	run-load-cpu run-load-cuda \
	run-unload-cpu run-unload-cuda run-overflow-cpu run-overflow-cuda \
	run-overflow-cuda-stress run-xformers-cpu run-xformers-cuda \
	check check-xformers clean

all: cpu cuda load-cpu load-cuda unload-cpu unload-cuda overflow-cpu overflow-cuda

cpu: $(BUILD_DIR)/detect-cpu

cuda: $(BUILD_DIR)/detect-cuda

opencl: $(BUILD_DIR)/detect-opencl

vulkan: $(BUILD_DIR)/detect-vulkan

opengl: $(BUILD_DIR)/detect-opengl

mesa: $(BUILD_DIR)/detect-mesa

load-cpu: $(BUILD_DIR)/load-cpu

load-cuda: $(BUILD_DIR)/load-cuda

unload-cpu: $(BUILD_DIR)/unload-cpu

unload-cuda: $(BUILD_DIR)/unload-cuda

overflow-cpu: $(BUILD_DIR)/overflow-cpu

overflow-cuda: $(BUILD_DIR)/overflow-cuda

xformers-cpu: $(BUILD_DIR)/xformers-load-unload-cpu-test

xformers-cuda: $(BUILD_DIR)/xformers-load-unload-cuda-test

xformers-cpu-gen: $(XFORMERS_IMAGE_SOURCES)
	$(CMAKE) -S $(XFORMERS_IMAGE_DIR) \
		-B $(BUILD_DIR)/xformers-image-cpu-cmake \
		-DXFORMERS_IMAGE_BACKEND=CPU \
		-DXFORMERS_IMAGE_OUTPUT_DIR=$(abspath $(BUILD_DIR))
	$(CMAKE) --build $(BUILD_DIR)/xformers-image-cpu-cmake \
		--target xformers-cpu-gen --parallel 4
	test -x $(BUILD_DIR)/xformers-cpu-gen

xformers-cuda-gen: $(XFORMERS_IMAGE_SOURCES)
	$(CMAKE) -S $(XFORMERS_IMAGE_DIR) \
		-B $(BUILD_DIR)/xformers-image-cuda-cmake \
		-DXFORMERS_IMAGE_BACKEND=CUDA \
		-DXFORMERS_IMAGE_OUTPUT_DIR=$(abspath $(BUILD_DIR))
	$(CMAKE) --build $(BUILD_DIR)/xformers-image-cuda-cmake \
		--target xformers-cuda-gen --parallel 4
	test -x $(BUILD_DIR)/xformers-cuda-gen

run-cpu: cpu
	$(BUILD_DIR)/detect-cpu

run-cuda: cuda
	$(BUILD_DIR)/detect-cuda

run-opencl: opencl
	$(BUILD_DIR)/detect-opencl

run-vulkan: vulkan
	$(BUILD_DIR)/detect-vulkan

run-opengl: opengl
	$(BUILD_DIR)/detect-opengl

run-mesa: mesa
	$(BUILD_DIR)/detect-mesa

run-load-cpu: load-cpu
	$(BUILD_DIR)/load-cpu

run-load-cuda: load-cuda
	$(BUILD_DIR)/load-cuda

run-unload-cpu: unload-cpu
	$(BUILD_DIR)/unload-cpu

run-unload-cuda: unload-cuda
	$(BUILD_DIR)/unload-cuda

run-overflow-cpu: overflow-cpu
	$(BUILD_DIR)/overflow-cpu

run-overflow-cuda: overflow-cuda
	$(BUILD_DIR)/overflow-cuda

run-overflow-cuda-stress: overflow-cuda
	$(BUILD_DIR)/overflow-cuda --stress

run-xformers-cpu: xformers-cpu
	$(BUILD_DIR)/xformers-load-unload-cpu-test

run-xformers-cuda: xformers-cuda
	$(BUILD_DIR)/xformers-load-unload-cuda-test

check: run-cpu run-cuda run-load-cpu run-load-cuda \
	run-unload-cpu run-unload-cuda run-overflow-cpu run-overflow-cuda

check-xformers: run-xformers-cpu run-xformers-cuda

$(BUILD_DIR)/detect-cpu: $(CPU_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BUILD_DIR)/detect-cuda: $(CUDA_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ $(CUDA_LDLIBS) -o $@

$(BUILD_DIR)/detect-opencl: $(OPENCL_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) \
		-DCL_TARGET_OPENCL_VERSION=120 -DEDCPP_DETECT_OPENCL=1 \
		$^ $(OPENCL_LDLIBS) -o $@

$(BUILD_DIR)/detect-vulkan: $(VULKAN_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) \
		-DEDCPP_DETECT_VULKAN=1 $^ $(VULKAN_LDLIBS) -o $@

$(BUILD_DIR)/detect-opengl: $(OPENGL_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) \
		-DEDCPP_DETECT_OPENGL=1 $^ $(OPENGL_LDLIBS) -o $@

$(BUILD_DIR)/detect-mesa: $(MESA_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) \
		-DEDCPP_DETECT_MESA=1 $^ $(MESA_LDLIBS) -o $@

$(BUILD_DIR)/load-cpu: $(LOAD_CPU_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BUILD_DIR)/load-cuda: $(LOAD_CUDA_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ $(CUDA_LDLIBS) -o $@

$(BUILD_DIR)/unload-cpu: $(UNLOAD_CPU_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BUILD_DIR)/unload-cuda: $(UNLOAD_CUDA_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ $(CUDA_LDLIBS) -o $@

$(BUILD_DIR)/overflow-cpu: $(CPU_OVERFLOW_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BUILD_DIR)/overflow-cuda: $(CUDA_OVERFLOW_SOURCES)
	@mkdir -p $(dir $@)
	$(NVCC) -O2 -std=c++17 $(INCLUDES) -DEDCPP_OVERFLOW_CUDA=1 $^ -o $@

$(BUILD_DIR)/xformers-load-unload-cpu-test: $(XFORMERS_CPU_CYCLE_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BUILD_DIR)/xformers-load-unload-cuda-test: $(XFORMERS_CUDA_CYCLE_SOURCES)
	@mkdir -p $(dir $@)
	$(NVCC) -O2 -std=c++17 $(CUDA_CPPFLAGS) $(INCLUDES) $^ -o $@

clean:
	$(RM) -r $(BUILD_DIR)
