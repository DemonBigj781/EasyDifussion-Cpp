cmake_minimum_required(VERSION 3.16)
project(sdkit)

if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

option(SDKIT_BUILD_TOOLS "Build sdkit tools" OFF)
option(SDKIT_BUILD_NATIVE_VISION "Build the isolated LibTorch YOLO sidecar" OFF)
option(SDKIT_BUILD_IMAGE_TOOLS "Build lightweight native image helpers" ON)
option(SDKIT_OPENGL_BACKEND "Build the experimental OpenGL/GLSL backend scaffold" OFF)
option(SDKIT_BUILD_LLAMA_RUNTIME
    "Build llama-server beside sdkit as an isolated GGML runtime" OFF)
set(SDKIT_LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH
    "Path to the vendored llama.cpp source")
set(SDKIT_LLAMA_BUILD_JOBS "4" CACHE STRING
    "Maximum parallel jobs used by the isolated llama.cpp build")
set(SDKIT_TORCH_CUDART "" CACHE FILEPATH
    "Optional libcudart used by the selected LibTorch distribution")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build ggml as a shared library")
set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "Build stable-diffusion as a shared library")
set(SD_BUILD_SHARED_GGML_LIB ON CACHE BOOL "Build ggml as a separate shared library")

if(MSVC)
    add_compile_options($<$<CONFIG:Debug>:/bigobj>)
endif()

# IntelLLVM reports several diagnostics from the vendored stable-diffusion.cpp
# and ggml SYCL sources that are benign for the oneAPI compile gate. Scope the
# compatibility suppressions to IntelLLVM so GCC/Clang/CUDA/ROCm builds retain
# their normal warning behavior. These can be removed incrementally as the
# corresponding upstream sources are modernized.
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-literal-operator>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-value>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-inconsistent-missing-override>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-absolute-value>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-#pragma-messages>"
        "$<$<COMPILE_LANGUAGE:C>:-Wno-#pragma-messages>"
    )
endif()

add_subdirectory(stable-diffusion.cpp)

file(GLOB_RECURSE ED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(sdkit ${ED_SRC})

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp)

# GCC 13 diagnoses two intentional/third-party cases in the CPU compile gate:
# ModelManager is deliberately hidden to avoid ELF interposition with the
# unrelated stable-diffusion.cpp ModelManager, while stb_image_resize.h uses
# legacy enum arithmetic. Keep those diagnostics scoped to the sdkit executable
# rather than weakening ModelManager visibility or modifying vendored STB code.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sdkit PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-attributes>"
        "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-enum-enum-conversion>"
    )
endif()

if(SDKIT_OPENGL_BACKEND)
    find_package(OpenGL REQUIRED)
    target_compile_definitions(sdkit PRIVATE SDKIT_ENABLE_OPENGL_BACKEND=1)
    target_link_libraries(sdkit PRIVATE OpenGL::GL)
    message(STATUS "OpenGL/GLSL Theory backend scaffold enabled (compile/link only)")
endif()

if(MSVC)
    target_compile_options(sdkit PRIVATE /wd4566)
endif()

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
    set_target_properties(sdkit PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

if(APPLE)
    set_target_properties(sdkit PROPERTIES
        BUILD_RPATH "@executable_path"
        INSTALL_RPATH "@executable_path"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

target_link_libraries(sdkit PRIVATE ggml)
target_link_libraries(sdkit PRIVATE stable-diffusion)
target_link_libraries(sdkit PRIVATE Threads::Threads)

# stable-diffusion.cpp and llama.cpp intentionally pin different ggml
# revisions. Linking both ggml libraries into one process would interpose
# identically named C symbols and corrupt either runtime. Build llama-server as
# part of the same native bundle, but keep its address space isolated.
if(SDKIT_BUILD_LLAMA_RUNTIME)
    if(NOT EXISTS "${SDKIT_LLAMA_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Vendored llama.cpp source not found: ${SDKIT_LLAMA_SOURCE_DIR}")
    endif()
    include(ExternalProject)
    set(SDKIT_LLAMA_BUILD_DIR "${CMAKE_BINARY_DIR}/llama-runtime")
    set(SDKIT_LLAMA_BACKEND_ARGS
        -DGGML_CUDA=${SD_CUDA}
        -DGGML_HIP=${SD_HIPBLAS}
        -DGGML_METAL=${SD_METAL}
        -DGGML_VULKAN=${SD_VULKAN}
        -DGGML_SYCL=${SD_SYCL}
        -DGGML_SYCL_F16=${GGML_SYCL_F16}
    )

    if(SD_HIPBLAS)
        if(GPU_TARGETS)
            list(APPEND SDKIT_LLAMA_BACKEND_ARGS -DGPU_TARGETS=${GPU_TARGETS})
        endif()
        if(AMDGPU_TARGETS)
            list(APPEND SDKIT_LLAMA_BACKEND_ARGS -DAMDGPU_TARGETS=${AMDGPU_TARGETS})
        endif()
        if(CMAKE_HIP_ARCHITECTURES)
            list(APPEND SDKIT_LLAMA_BACKEND_ARGS
                -DCMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES})
        endif()
    endif()

    if(SD_SYCL)
        list(APPEND SDKIT_LLAMA_BACKEND_ARGS
            -DGGML_SYCL_TARGET=${GGML_SYCL_TARGET}
            -DGGML_SYCL_DNN=${GGML_SYCL_DNN}
            -DGGML_SYCL_SUPPORT_LEVEL_ZERO=${GGML_SYCL_SUPPORT_LEVEL_ZERO}
        )
        if(GGML_SYCL_DEVICE_ARCH)
            list(APPEND SDKIT_LLAMA_BACKEND_ARGS
                -DGGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH})
        endif()
    endif()

    # Theory-only Intel CPU path: if the diffusion GGML tree is using BLAS,
    # keep the isolated llama runtime on the same BLAS vendor policy. This is
    # what lets oneAPI/Xeon builds use oneMKL consistently in both processes.
    if(GGML_BLAS)
        list(APPEND SDKIT_LLAMA_BACKEND_ARGS
            -DGGML_BLAS=ON
            -DGGML_BLAS_VENDOR=${GGML_BLAS_VENDOR}
            -DGGML_NATIVE=${GGML_NATIVE}
        )
    endif()

    ExternalProject_Add(sdkit-llama-runtime
        SOURCE_DIR "${SDKIT_LLAMA_SOURCE_DIR}"
        BINARY_DIR "${SDKIT_LLAMA_BUILD_DIR}"
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DBUILD_SHARED_LIBS=OFF
            -DLLAMA_CURL=OFF
            -DLLAMA_BUILD_TESTS=OFF
            -DLLAMA_BUILD_TOOLS=ON
            -DLLAMA_BUILD_EXAMPLES=OFF
            -DLLAMA_BUILD_SERVER=ON
            -DLLAMA_BUILD_APP=OFF
            -DLLAMA_BUILD_UI=OFF
            -DLLAMA_USE_PREBUILT_UI=OFF
            ${SDKIT_LLAMA_BACKEND_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server
            --parallel ${SDKIT_LLAMA_BUILD_JOBS}
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS
            "${SDKIT_LLAMA_BUILD_DIR}/bin/llama-server${CMAKE_EXECUTABLE_SUFFIX}"
    )
    add_dependencies(sdkit sdkit-llama-runtime)
    add_custom_command(TARGET sdkit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDKIT_LLAMA_BUILD_DIR}/bin/llama-server${CMAKE_EXECUTABLE_SUFFIX}"
            "$<TARGET_FILE_DIR:sdkit>"
        COMMENT "Bundling the isolated llama.cpp runtime with sdkit")

    set(SDKIT_LLAMA_CONVERTER_DIR "$<TARGET_FILE_DIR:sdkit>/llama-converters")
    add_custom_command(TARGET sdkit POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SDKIT_LLAMA_CONVERTER_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDKIT_LLAMA_SOURCE_DIR}/convert_hf_to_gguf.py"
            "${SDKIT_LLAMA_SOURCE_DIR}/convert_hf_to_gguf_update.py"
            "${SDKIT_LLAMA_SOURCE_DIR}/convert_llama_ggml_to_gguf.py"
            "${SDKIT_LLAMA_SOURCE_DIR}/convert_lora_to_gguf.py"
            "${SDKIT_LLAMA_CONVERTER_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SDKIT_LLAMA_SOURCE_DIR}/conversion"
            "${SDKIT_LLAMA_CONVERTER_DIR}/conversion"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SDKIT_LLAMA_SOURCE_DIR}/gguf-py"
            "${SDKIT_LLAMA_CONVERTER_DIR}/gguf-py"
        COMMENT "Bundling llama.cpp GGUF conversion tools with sdkit")
endif()

if(WIN32)
    target_link_libraries(sdkit PRIVATE ws2_32)
    add_compile_definitions(_WIN32_WINNT=0x0601)
endif()

add_custom_command(TARGET sdkit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:stable-diffusion>
        $<TARGET_FILE:ggml>
        $<TARGET_FILE_DIR:sdkit>
)

if(SDKIT_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(SDKIT_BUILD_NATIVE_VISION)
    find_package(Torch REQUIRED)
    if(SDKIT_TORCH_CUDART)
        if(NOT EXISTS "${SDKIT_TORCH_CUDART}")
            message(FATAL_ERROR "SDKIT_TORCH_CUDART does not exist: ${SDKIT_TORCH_CUDART}")
        endif()
        set(TORCH_CUDART_LIBRARY "${SDKIT_TORCH_CUDART}" CACHE FILEPATH
            "CUDA runtime used by TorchConfig" FORCE)
    endif()
    add_subdirectory(native_vision)
endif()
