cmake_minimum_required(VERSION 3.16)
project(sdkit)

# Release bundles are copied directly into Easy Diffusion's backend directory.
# Keep every shared-library lookup relative to that bundle instead of embedding
# an absolute developer build path.
if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable versioned sonames on platforms that support it
set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

# Option to build tools
option(SDKIT_BUILD_TOOLS "Build sdkit tools" OFF)
option(SDKIT_BUILD_TESTS "Build sdkit regression tests" OFF)
option(SDKIT_BUILD_NATIVE_VISION "Build the isolated LibTorch YOLO sidecar" OFF)
option(SDKIT_BUILD_SPRITE_GPT "Build the isolated native Sprite-GPT sidecar" OFF)
option(SDKIT_BUILD_IMAGE_TOOLS "Build lightweight native image helpers" ON)
option(SDKIT_BUILD_SD15_TRAINER "Build the native C++ SD 1.5 LoRA trainer" ON)
file(GLOB SDKIT_BUNDLED_ORT_LIBS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*")
file(GLOB SDKIT_BUNDLED_ORT_PROVIDER_LIBS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/lib/python*/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so*")
set(_SDKIT_WD14_DEFAULT OFF)
if(SDKIT_BUNDLED_ORT_LIBS)
    set(_SDKIT_WD14_DEFAULT ON)
endif()
option(SDKIT_BUILD_WD14_TAGGER "Build the native WD14 ONNX tagger sidecar" ${_SDKIT_WD14_DEFAULT})
set(SDKIT_ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "ONNX Runtime shared library used by the native WD14 tagger")
option(SDKIT_BUILD_LLAMA_RUNTIME
    "Build llama-server beside sdkit as an isolated GGML runtime" OFF)
set(SDKIT_LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH
    "Path to the vendored llama.cpp source")
set(SDKIT_LLAMA_BUILD_JOBS "4" CACHE STRING
    "Maximum parallel jobs used by the isolated llama.cpp build")
set(SDKIT_TORCH_CUDART "" CACHE FILEPATH
    "Optional libcudart used by the selected LibTorch distribution")

# Define output directories for all build artifacts
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Build the C++ page renderer alongside SDKit for the FastAPI /cpp-ui routes.
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../UI.cpp/Shared_Core"
    "${CMAKE_CURRENT_BINARY_DIR}/ui-cpp"
)

# Set flags for shared library builds
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build ggml as a shared library")
set(SD_BUILD_SHARED_LIBS ON CACHE BOOL "Build stable-diffusion as a shared library")
set(SD_BUILD_SHARED_GGML_LIB ON CACHE BOOL "Build ggml as a separate shared library")

# Enable /bigobj for MSVC to handle large object files (needed for debug builds)
if(MSVC)
    add_compile_options($<$<CONFIG:Debug>:/bigobj>)
endif()

# Include stable-diffusion.cpp
add_subdirectory(stable-diffusion.cpp)

if(SDKIT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(stable-diffusion.cpp/tests)
endif()

# Add the sdkit binary
file(GLOB_RECURSE ED_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(sdkit ${ED_SRC})

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

target_include_directories(sdkit PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp)
add_dependencies(sdkit easy-diffusion-ui-render)

add_custom_command(TARGET sdkit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../UI.cpp/Pages/assets"
        "$<TARGET_FILE_DIR:sdkit>/cpp-ui-assets"
    COMMENT "Bundling C++ UI assets with SDKit")

add_custom_command(TARGET sdkit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/../UI.cpp/Pages/src/Plugin/plugin_scripts"
        "$<TARGET_FILE_DIR:sdkit>/cpp-ui-scripts"
    COMMENT "Bundling C++ UI scripts with SDKit")

# Suppress Unicode character warnings from Crow library headers on Windows
if(MSVC)
    target_compile_options(sdkit PRIVATE /wd4566)
endif()

# Find Threads package for multithreading support
find_package(Threads REQUIRED)

if(SDKIT_BUILD_TESTS)
    add_executable(generation-error-test tests/test_generation_error.cpp src/logging.cpp)
    target_include_directories(generation-error-test PRIVATE include stable-diffusion.cpp/include)
    target_link_libraries(generation-error-test PRIVATE Threads::Threads)
    add_test(NAME generation-error-test COMMAND generation-error-test)
endif()

# Set RPATH for Linux/Unix to find shared libraries in the same directory as the executable
if(UNIX AND NOT APPLE)
    set_target_properties(sdkit PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

# For macOS, use @executable_path
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

if(SDKIT_BUILD_SD15_TRAINER)
    add_executable(sdkit-sd15-trainer tools/sd15_lora_train.cpp)
    target_include_directories(sdkit-sd15-trainer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/src
        ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/thirdparty)
    # The parent executable includes sdkit headers which rely on the enlarged
    # name size configured by stable-diffusion.cpp's build.
    target_compile_definitions(sdkit-sd15-trainer PRIVATE GGML_MAX_NAME=160)
    target_link_libraries(sdkit-sd15-trainer PRIVATE stable-diffusion ggml Threads::Threads)
    if(SDKIT_BUILD_TESTS)
        add_executable(sd15-epoch-sampling-test tests/sd15_epoch_sampling_test.cpp)
        add_test(NAME sd15-epoch-sampling COMMAND sd15-epoch-sampling-test)
        add_executable(sd15-training-schedule-test tests/sd15_training_schedule_test.cpp)
        add_test(NAME sd15-training-schedule COMMAND sd15-training-schedule-test)
        add_executable(sd15-lora-export-test tests/sd15_lora_export_test.cpp)
        target_include_directories(sd15-lora-export-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/thirdparty ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/src)
        target_compile_definitions(sd15-lora-export-test PRIVATE GGML_MAX_NAME=160)
        target_link_libraries(sd15-lora-export-test PRIVATE stable-diffusion ggml ggml-cpu)
        add_test(NAME sd15-lora-export COMMAND sd15-lora-export-test)
        add_executable(text-encoder-optimizer-test tests/text_encoder_optimizer_test.cpp)
        target_compile_definitions(text-encoder-optimizer-test PRIVATE GGML_MAX_NAME=160)
        target_link_libraries(text-encoder-optimizer-test PRIVATE ggml ggml-cpu)
        add_test(NAME text-encoder-optimizer-test COMMAND text-encoder-optimizer-test)
        add_executable(sd15-backward-math-test tests/backward_math_test.cpp)
        target_compile_definitions(sd15-backward-math-test PRIVATE GGML_MAX_NAME=160)
        target_link_libraries(sd15-backward-math-test PRIVATE ggml ggml-cpu)
        add_test(NAME sd15-backward-math-cpu COMMAND sd15-backward-math-test)
        add_executable(optimizer-failure-fixture tests/optimizer_failure_fixture.cpp)
        target_compile_definitions(optimizer-failure-fixture PRIVATE GGML_MAX_NAME=160)
        target_link_libraries(optimizer-failure-fixture PRIVATE ggml ggml-cpu)
        add_test(NAME optimizer-compute-failure
            COMMAND ${CMAKE_COMMAND} -DTEST_EXE=$<TARGET_FILE:optimizer-failure-fixture>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/assert_optimizer_failure.cmake)
    endif()
    add_dependencies(sdkit sdkit-sd15-trainer)
    if(UNIX AND NOT APPLE)
        set_target_properties(sdkit-sd15-trainer PROPERTIES
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    endif()
endif()

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
        -DGGML_SYCL_F16=${SD_SYCL}
    )
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

    # The upstream converters are Python programs and cannot be linked into
    # stable-diffusion.cpp's C++ process. Ship them in the same native bundle,
    # together with their local conversion modules, so the sdkit3 tool layer
    # can launch them through Easy Diffusion's main Python 3.13 environment.
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

# Platform-specific libraries
if(WIN32)
    target_link_libraries(sdkit PRIVATE ws2_32)
    # Target Windows 7 and later (0x0601). This sets the _WIN32_WINNT macro
    # for all targets processed by this CMakeLists.
    add_compile_definitions(_WIN32_WINNT=0x0601)
endif()

# Copy the necessary DLLs to the runtime output directory
add_custom_command(TARGET sdkit POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:stable-diffusion>
        $<TARGET_FILE:ggml>
        $<TARGET_FILE_DIR:sdkit>
)

# Build tools if enabled
if(SDKIT_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# LibTorch sidecars remain separate processes so their runtime and model
# allocations never increase the diffusion server's steady-state memory use.
if(SDKIT_BUILD_NATIVE_VISION OR SDKIT_BUILD_SPRITE_GPT)
    find_package(Torch REQUIRED)
    # Python LibTorch wheels are built against the CUDA runtime bundled with
    # the wheel.  Some systems also have an older /usr/lib CUDA runtime, which
    # CMake may otherwise select and fail at link time on versioned symbols.
    if(SDKIT_TORCH_CUDART)
        if(NOT EXISTS "${SDKIT_TORCH_CUDART}")
            message(FATAL_ERROR "SDKIT_TORCH_CUDART does not exist: ${SDKIT_TORCH_CUDART}")
        endif()
        if(TARGET CUDA::cudart)
            set_target_properties(CUDA::cudart PROPERTIES
                IMPORTED_LOCATION "${SDKIT_TORCH_CUDART}"
                IMPORTED_LOCATION_RELEASE "${SDKIT_TORCH_CUDART}")
        endif()
    endif()
endif()

# Ultralytics-compatible models are exported once to TorchScript; this
# executable contains only clean C++ image preparation, decoding, and NMS.
if(SDKIT_BUILD_NATIVE_VISION)
    add_executable(sdkit-vision tools/native_vision.cpp)
    target_include_directories(sdkit-vision PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/thirdparty)
    target_link_libraries(sdkit-vision PRIVATE ${TORCH_LIBRARIES})
    if(SDKIT_TORCH_CUDART)
        target_link_libraries(sdkit-vision PRIVATE "${SDKIT_TORCH_CUDART}")
    endif()
    if(UNIX AND NOT APPLE)
        set_target_properties(sdkit-vision PROPERTIES
            BUILD_RPATH_USE_ORIGIN TRUE
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    endif()
endif()

# Sprite-GPT uses LibTorch for native inference and rectified-flow training.
# Safetensors are used for exported training weights; OpenCV is not required by
# this target because the bundled stb image code handles its PNG/JPEG inputs.
if(SDKIT_BUILD_SPRITE_GPT)
    find_package(ICU REQUIRED COMPONENTS uc)
    add_executable(sdkit-sprite-gpt tools/sprite_gpt.cpp)
    target_include_directories(sdkit-sprite-gpt PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/thirdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
    target_link_libraries(sdkit-sprite-gpt PRIVATE ${TORCH_LIBRARIES} ICU::uc)
    if(SDKIT_TORCH_CUDART)
        target_link_libraries(sdkit-sprite-gpt PRIVATE "${SDKIT_TORCH_CUDART}")
    endif()
    if(UNIX AND NOT APPLE)
        set_target_properties(sdkit-sprite-gpt PROPERTIES
            BUILD_RPATH_USE_ORIGIN TRUE
            BUILD_RPATH "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
            INSTALL_RPATH_USE_LINK_PATH FALSE)
    endif()
    add_dependencies(sdkit sdkit-sprite-gpt)
endif()

# Lightweight helpers intentionally have no LibTorch/OpenCV dependency.  The
# text-mask process exits after each request and therefore cannot add to the
# diffusion server's steady-state RAM use.
if(SDKIT_BUILD_IMAGE_TOOLS)
    add_executable(sdkit-image-tools tools/native_image_tools.cpp)
    target_include_directories(sdkit-image-tools PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/thirdparty)
endif()

# The tagger runs as a persistent sidecar so one loaded ONNX session serves all
# images in an autotag batch. Its public headers are pinned to ONNX Runtime 1.23.2.
if(SDKIT_BUILD_WD14_TAGGER)
    if(NOT SDKIT_ONNXRUNTIME_LIBRARY)
        if(SDKIT_BUNDLED_ORT_LIBS)
            list(GET SDKIT_BUNDLED_ORT_LIBS 0 SDKIT_ONNXRUNTIME_LIBRARY)
        else()
            find_library(SDKIT_ONNXRUNTIME_LIBRARY NAMES onnxruntime
                HINTS "$ENV{ONNXRUNTIME_ROOT}/lib" "$ENV{ONNXRUNTIME_ROOT}/lib64")
        endif()
    endif()
    if(NOT EXISTS "${SDKIT_ONNXRUNTIME_LIBRARY}")
        message(FATAL_ERROR "SDKIT_BUILD_WD14_TAGGER requires ONNX Runtime 1.23.2; set SDKIT_ONNXRUNTIME_LIBRARY")
    endif()
    add_executable(sdkit-wd14-tagger tools/wd14_tagger.cpp)
    add_dependencies(sdkit sdkit-wd14-tagger)
    target_include_directories(sdkit-wd14-tagger PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include)
    target_link_libraries(sdkit-wd14-tagger PRIVATE "${SDKIT_ONNXRUNTIME_LIBRARY}")
    if(UNIX AND NOT APPLE)
        set_target_properties(sdkit-wd14-tagger PROPERTIES
            BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    endif()
    add_custom_command(TARGET sdkit-wd14-tagger POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SDKIT_ONNXRUNTIME_LIBRARY}" "$<TARGET_FILE_DIR:sdkit-wd14-tagger>"
        COMMENT "Bundling ONNX Runtime with the native WD14 tagger")
    foreach(SDKIT_ORT_PROVIDER_LIBRARY IN LISTS SDKIT_BUNDLED_ORT_PROVIDER_LIBS)
        add_custom_command(TARGET sdkit-wd14-tagger POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${SDKIT_ORT_PROVIDER_LIBRARY}" "$<TARGET_FILE_DIR:sdkit-wd14-tagger>")
    endforeach()
    if(UNIX AND NOT APPLE)
        get_filename_component(SDKIT_ORT_LIBRARY_NAME "${SDKIT_ONNXRUNTIME_LIBRARY}" NAME)
        add_custom_command(TARGET sdkit-wd14-tagger POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f "$<TARGET_FILE_DIR:sdkit-wd14-tagger>/libonnxruntime.so.1"
            COMMAND ${CMAKE_COMMAND} -E create_symlink "${SDKIT_ORT_LIBRARY_NAME}"
                "$<TARGET_FILE_DIR:sdkit-wd14-tagger>/libonnxruntime.so.1")
    endif()
endif()
