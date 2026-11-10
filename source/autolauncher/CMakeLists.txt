cmake_minimum_required(VERSION 3.16)
project(edcpp_autolauncher LANGUAGES C CXX)

if(EDCPP_COSMOPOLITAN AND NOT CMAKE_CXX_OUTPUT_EXTENSION STREQUAL ".o")
    message(FATAL_ERROR "Use the supplied Cosmopolitan toolchain and .o rules")
endif()
set(_hash "${CMAKE_CURRENT_LIST_DIR}/../llama.cpp/vendor/hash")
add_library(launcher_sha256 STATIC "${_hash}/sha256/sha256.c")
target_include_directories(launcher_sha256 PUBLIC "${_hash}" "${_hash}/sha256")
add_executable(easy-diffusion-launcher launcher.cpp process.cpp platform.cpp)
target_compile_features(easy-diffusion-launcher PRIVATE cxx_std_17)
target_compile_definitions(easy-diffusion-launcher PRIVATE _GNU_SOURCE)
target_compile_options(easy-diffusion-launcher PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(easy-diffusion-launcher PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../sdkit3-port-source/stable-diffusion.cpp/thirdparty")
target_link_libraries(easy-diffusion-launcher PRIVATE launcher_sha256)
set_target_properties(easy-diffusion-launcher PROPERTIES
    CXX_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(EDCPP_COSMOPOLITAN)
    set_target_properties(easy-diffusion-launcher PROPERTIES OUTPUT_NAME easy-diffusion SUFFIX .com)
endif()

option(EDCPP_BUILD_CUDA_PROBE "Build a native CUDA-driver compatibility probe" OFF)
if(EDCPP_BUILD_CUDA_PROBE)
    if(EDCPP_COSMOPOLITAN)
        message(FATAL_ERROR "Build cuda-probe separately using the host/native toolchain")
    endif()
    add_executable(cuda-probe cuda_probe.cpp)
    target_compile_features(cuda-probe PRIVATE cxx_std_17)
    target_link_libraries(cuda-probe PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(cuda-probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

set(EDCPP_AUTOLAUNCHER_BUNDLE_ROOT "" CACHE PATH "Already staged application bundle")
set(EDCPP_AUTOLAUNCHER_BUNDLE_SPEC "" CACHE FILEPATH "Packaging specification JSON")
if(EDCPP_AUTOLAUNCHER_BUNDLE_ROOT OR EDCPP_AUTOLAUNCHER_BUNDLE_SPEC)
    if(NOT EDCPP_AUTOLAUNCHER_BUNDLE_ROOT OR NOT EDCPP_AUTOLAUNCHER_BUNDLE_SPEC)
        message(FATAL_ERROR "Packaging needs both BUNDLE_ROOT and BUNDLE_SPEC")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    # An explicit always-run packaging target also updates hashes when workers
    # change but the launcher itself did not relink.
    add_custom_target(autolauncher-package
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/../../scripts/package_autolauncher.py"
            --root "${EDCPP_AUTOLAUNCHER_BUNDLE_ROOT}"
            --spec "${EDCPP_AUTOLAUNCHER_BUNDLE_SPEC}"
            --launcher "$<TARGET_FILE:easy-diffusion-launcher>"
            --replace
        DEPENDS easy-diffusion-launcher VERBATIM)
endif()
