cmake_minimum_required(VERSION 3.16)
project(attention-image-test LANGUAGES C CXX)

set(XFORMERS_IMAGE_BACKEND "CPU" CACHE STRING "CPU or CUDA")
set_property(CACHE XFORMERS_IMAGE_BACKEND PROPERTY STRINGS CPU CUDA)
set(XFORMERS_IMAGE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin" CACHE PATH
    "Directory for the image-generator executable")

string(TOUPPER "${XFORMERS_IMAGE_BACKEND}" XFORMERS_IMAGE_BACKEND)
if(NOT XFORMERS_IMAGE_BACKEND STREQUAL "CPU" AND
   NOT XFORMERS_IMAGE_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR "XFORMERS_IMAGE_BACKEND must be CPU or CUDA")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)
set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SD_WEBP OFF CACHE BOOL "" FORCE)
set(SD_WEBM OFF CACHE BOOL "" FORCE)
if(XFORMERS_IMAGE_BACKEND STREQUAL "CUDA")
    set(SD_CUDA ON CACHE BOOL "" FORCE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # The CUDA workflow configures without a visible GPU. The Common
        # attention kernels support Pascal, but the complete ggml-CUDA target also
        # compiles the SM80 SageAttention translation, whose CUDA primitives
        # cannot be compiled for pre-Ampere architectures. Emit SM80 PTX so the
        # standalone integration binary builds without device auto-detection
        # and remains JIT-compatible with newer NVIDIA devices.
        set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING
            "CUDA architectures for the standalone xFormers image test")
    endif()
else()
    set(SD_CUDA OFF CACHE BOOL "" FORCE)
endif()

set(SD_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../sdkit3-port-source/stable-diffusion.cpp")
add_subdirectory("${SD_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/stable-diffusion")

string(TOLOWER "${XFORMERS_IMAGE_BACKEND}" backend_lower)
set(target "xformers-${backend_lower}-gen")
add_executable("${target}"
    xformers-image-gen.cpp
    "xformers-${backend_lower}-gen.cpp")
target_compile_features("${target}" PRIVATE cxx_std_17)
target_include_directories("${target}" PRIVATE "${SD_SOURCE_DIR}/include")
target_link_libraries("${target}" PRIVATE stable-diffusion)
if(XFORMERS_IMAGE_BACKEND STREQUAL "CUDA")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries("${target}" PRIVATE CUDA::cudart)
endif()
set_target_properties("${target}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${XFORMERS_IMAGE_OUTPUT_DIR}")

if(XFORMERS_IMAGE_BACKEND STREQUAL "CUDA")
    set(FLASH_IMAGE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../../flash/generate-image")
    add_executable(flash-cuda-gen
        xformers-image-gen.cpp
        "${FLASH_IMAGE_DIR}/flash-cuda-gen.cpp")
    target_compile_features(flash-cuda-gen PRIVATE cxx_std_17)
    target_include_directories(flash-cuda-gen PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SD_SOURCE_DIR}/include")
    target_link_libraries(flash-cuda-gen PRIVATE
        stable-diffusion CUDA::cudart)
    set_target_properties(flash-cuda-gen PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${XFORMERS_IMAGE_OUTPUT_DIR}")
endif()
