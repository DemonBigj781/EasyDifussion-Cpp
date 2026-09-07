cmake_minimum_required(VERSION 3.16)
project(edcpp-sdkit3-ggml-bridge LANGUAGES CXX CUDA)

set(EDCPP_API_ROOT "" CACHE PATH "Path to source/API.cpp")
set(EDCPP_GGML_ROOT "" CACHE PATH "Path to stable-diffusion.cpp/ggml")

if(NOT IS_DIRECTORY "${EDCPP_API_ROOT}")
    message(FATAL_ERROR "EDCPP_API_ROOT must name the API.cpp source directory")
endif()
if(NOT IS_DIRECTORY "${EDCPP_GGML_ROOT}/src/ggml-cuda")
    message(FATAL_ERROR "EDCPP_GGML_ROOT must name the compatible GGML source directory")
endif()

add_library(edcpp-sdkit3-ggml-adapters OBJECT
    ggml-cuda/flash-attention.cu
    ggml-cuda/xformers-attention.cu)
target_compile_features(edcpp-sdkit3-ggml-adapters PRIVATE cxx_std_17)
target_include_directories(edcpp-sdkit3-ggml-adapters PRIVATE
    "${EDCPP_API_ROOT}"
    "${EDCPP_GGML_ROOT}/include"
    "${EDCPP_GGML_ROOT}/src"
    "${EDCPP_GGML_ROOT}/src/ggml-cuda")

# The standalone project verifies the adapter boundary compiles. Production
# links these same sources into ggml-cuda so they share its internal symbols.
set_target_properties(edcpp-sdkit3-ggml-adapters PROPERTIES
    CUDA_ARCHITECTURES 80)
