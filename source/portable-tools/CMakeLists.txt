cmake_minimum_required(VERSION 3.16)
project(edcpp_portable_tools LANGUAGES C CXX)

# Standalone compatibility trial: never add_subdirectory() SDKIT3 here. That
# would import shared GGML libraries, GPU SDKs, and compiler-forwarding rules.
if(BUILD_SHARED_LIBS)
    message(FATAL_ERROR "The portable-tools trial requires BUILD_SHARED_LIBS=OFF")
endif()
if(EDCPP_COSMOPOLITAN AND NOT CMAKE_CXX_OUTPUT_EXTENSION STREQUAL ".o")
    message(FATAL_ERROR "Cosmopolitan requires the .o object rules from the supplied toolchain")
endif()

set(_sdkit_source "${CMAKE_CURRENT_LIST_DIR}/../sdkit3-port-source")
add_executable(sdkit-image-tools "${_sdkit_source}/tools/native_image_tools.cpp")
target_include_directories(sdkit-image-tools PRIVATE
    "${_sdkit_source}/stable-diffusion.cpp/thirdparty")
target_compile_features(sdkit-image-tools PRIVATE cxx_std_20)
set_target_properties(sdkit-image-tools PROPERTIES
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(EDCPP_COSMOPOLITAN)
    # No host-native ISA flags: this helper must run before accelerator choice.
    # The fat compiler produces an APE plus .dbg/.aarch64.elf debugging outputs.
    set_target_properties(sdkit-image-tools PROPERTIES SUFFIX ".com")
endif()

# Deliberately no install, application startup, GPU probing, or runtime tests.
# Successful compilation is not a Theory runtime-support claim.
