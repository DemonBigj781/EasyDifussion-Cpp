CXX ?= c++
NVCC ?= nvcc
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS ?=
CUDA_CPPFLAGS ?= -DEDCPP_DETECT_CUDA=1 -DEDCPP_LOAD_CUDA=1 -DEDCPP_UNLOAD_CUDA=1
CUDA_LDLIBS ?= -lcudart

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

.PHONY: all cpu cuda load-cpu load-cuda unload-cpu unload-cuda overflow-cpu overflow-cuda \
	run-cpu run-cuda run-load-cpu run-load-cuda \
	run-unload-cpu run-unload-cuda run-overflow-cpu run-overflow-cuda \
	run-overflow-cuda-stress check clean

all: cpu cuda load-cpu load-cuda unload-cpu unload-cuda overflow-cpu overflow-cuda

cpu: $(BUILD_DIR)/detect-cpu

cuda: $(BUILD_DIR)/detect-cuda

load-cpu: $(BUILD_DIR)/load-cpu

load-cuda: $(BUILD_DIR)/load-cuda

unload-cpu: $(BUILD_DIR)/unload-cpu

unload-cuda: $(BUILD_DIR)/unload-cuda

overflow-cpu: $(BUILD_DIR)/overflow-cpu

overflow-cuda: $(BUILD_DIR)/overflow-cuda

run-cpu: cpu
	$(BUILD_DIR)/detect-cpu

run-cuda: cuda
	$(BUILD_DIR)/detect-cuda

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

check: run-cpu run-cuda run-load-cpu run-load-cuda \
	run-unload-cpu run-unload-cuda run-overflow-cpu run-overflow-cuda

$(BUILD_DIR)/detect-cpu: $(CPU_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(BUILD_DIR)/detect-cuda: $(CUDA_SOURCES)
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CXXFLAGS) $(INCLUDES) $^ $(CUDA_LDLIBS) -o $@

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

clean:
	$(RM) -r $(BUILD_DIR)
