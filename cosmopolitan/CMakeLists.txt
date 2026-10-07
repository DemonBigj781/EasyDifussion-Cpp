cmake_minimum_required(VERSION 3.24)
project(easy_diffusion_cosmopolitan LANGUAGES C CXX ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
# SD's model implementation and compiled vocabularies can each use several
# GiB during compilation. Keep these and native application units serialized.
set_property(GLOBAL PROPERTY JOB_POOLS cosmo_large_cpp=1)
set(COSMO_SOURCE_ROOT "" CACHE PATH "Prepared application source")
set(COSMO_SHARED_GGML "" CACHE PATH "Prepared shared GGML source")
option(COSMO_BUILD_APP "Build the combined application" ON)
if(NOT EXISTS "${COSMO_SOURCE_ROOT}/llama.cpp/CMakeLists.txt")
    message(FATAL_ERROR "Run cosmopolitan/build.py to prepare source and tools")
endif()
if(NOT EXISTS "${COSMO_SHARED_GGML}/CMakeLists.txt")
    message(FATAL_ERROR "The authoritative shared GGML tree is not prepared")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_CPU ON CACHE BOOL "" FORCE)
foreach(option IN ITEMS
        GGML_BACKEND_DL GGML_NATIVE GGML_OPENMP GGML_OPENMP_FETCH GGML_LTO GGML_CCACHE
        GGML_CPU_ALL_VARIANTS GGML_CPU_REPACK GGML_CPU_HBM GGML_CPU_KLEIDIAI GGML_LLAMAFILE
        GGML_SSE42 GGML_AVX GGML_AVX2 GGML_AVX_VNNI GGML_BMI2 GGML_FMA GGML_F16C
        GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16
        GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16 GGML_BLAS GGML_ACCELERATE
        GGML_CUDA GGML_HIP GGML_METAL GGML_VULKAN GGML_OPENCL
        GGML_RPC GGML_SYCL GGML_MUSA GGML_BUILD_TESTS GGML_BUILD_EXAMPLES
        LLAMA_BUILD_COMMON LLAMA_BUILD_TOOLS LLAMA_BUILD_EXAMPLES LLAMA_BUILD_TESTS
        LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_BUILD_UI LLAMA_BUILD_MTMD
        LLAMA_OPENSSL LLAMA_SUBPROCESS LLAMA_LLGUIDANCE LLAMA_USE_SYSTEM_GGML
        SD_BUILD_EXAMPLES SD_BUILD_SHARED_LIBS SD_BUILD_SHARED_GGML_LIB SD_USE_SYSTEM_GGML
        SD_WEBP SD_WEBM SD_CUDA SD_HIPBLAS SD_METAL SD_VULKAN SD_OPENCL SD_SYCL SD_MUSA)
    set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
set(GGML_WEBGPU ON CACHE BOOL "Use the shared GGML WebGPU backend" FORCE)
set(GGML_WEBGPU_COSMO ON CACHE BOOL "Use the pinned static Cosmopolitan WebGPU stack" FORCE)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/software-webgpu.cmake")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/backend" backend)
add_compile_definitions(GGML_MAX_NAME=160)
add_subdirectory("${COSMO_SHARED_GGML}" shared-ggml)
target_compile_definitions(ggml PUBLIC GGML_MAX_NAME=160)
target_compile_definitions(ggml PRIVATE GGML_COSMO_STATIC_ONLY)
target_compile_definitions(ggml-base PUBLIC GGML_MAX_NAME=160)
add_subdirectory("${COSMO_SOURCE_ROOT}/llama.cpp" llama EXCLUDE_FROM_ALL)
set(SDKIT_SOURCE "${COSMO_SOURCE_ROOT}/sdkit3-port-source")
set(SD_SOURCE "${SDKIT_SOURCE}/stable-diffusion.cpp")
add_subdirectory("${SD_SOURCE}" stable-diffusion EXCLUDE_FROM_ALL)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/split-vocabulary.cmake")
set_property(TARGET stable-diffusion PROPERTY JOB_POOL_COMPILE cosmo_large_cpp)
add_subdirectory("${COSMO_SOURCE_ROOT}/UI.cpp/Shared_Core" ui EXCLUDE_FROM_ALL)

if(COSMO_BUILD_APP)
    file(GLOB SDKIT_SOURCES CONFIGURE_DEPENDS "${SDKIT_SOURCE}/src/*.cpp")
    list(REMOVE_ITEM SDKIT_SOURCES "${SDKIT_SOURCE}/src/main.cpp")
    add_library(cosmo-sdkit STATIC ${SDKIT_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/src/sdkit_entry.cpp")
    set_property(TARGET cosmo-sdkit PROPERTY JOB_POOL_COMPILE cosmo_large_cpp)
    target_compile_features(cosmo-sdkit PRIVATE cxx_std_20)
    target_include_directories(cosmo-sdkit PUBLIC
        "${SDKIT_SOURCE}/include" "${SDKIT_SOURCE}/third_party"
        "${SD_SOURCE}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_compile_definitions(cosmo-sdkit PUBLIC
        ASIO_DISABLE_EPOLL ASIO_DISABLE_EVENTFD ASIO_DISABLE_TIMERFD ASIO_DISABLE_SERIAL_PORT)
    target_compile_definitions(cosmo-sdkit PRIVATE COSMO_SDKIT_MAIN_SOURCE="${SDKIT_SOURCE}/src/main.cpp")
    target_link_libraries(cosmo-sdkit PUBLIC stable-diffusion ggml easy-diffusion-ui cosmo-ggml-webgpu-adapter)
    file(GLOB FRONTEND_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
    list(REMOVE_ITEM FRONTEND_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/sdkit_entry.cpp")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/math_compat.c"
        PROPERTIES COMPILE_OPTIONS "-fno-builtin")
    add_executable(easy-diffusion ${FRONTEND_SOURCES}
        "${CMAKE_CURRENT_SOURCE_DIR}/shared-ggml/compat_test.c")
    set_target_properties(easy-diffusion PROPERTIES
        OUTPUT_NAME "easy-diffusion" SUFFIX ".com.dbg" LINKER_LANGUAGE CXX
        JOB_POOL_COMPILE cosmo_large_cpp)
    target_include_directories(easy-diffusion PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src" "${SD_SOURCE}/src" "${SD_SOURCE}/thirdparty"
        "${SD_SOURCE}" "${COSMO_SHARED_GGML}/src")
    target_compile_definitions(easy-diffusion PRIVATE
        COSMO_WEBGPU_BACKEND=1
        COSMO_TRAINER_SOURCE="${SDKIT_SOURCE}/tools/sd15_lora_train.cpp"
        COSMO_BACKWARD_TEST_SOURCE="${SDKIT_SOURCE}/tests/backward_math_test.cpp"
        COSMO_OPTIMIZER_TEST_SOURCE="${SDKIT_SOURCE}/tests/text_encoder_optimizer_test.cpp")
    target_link_libraries(easy-diffusion PRIVATE llama cosmo-sdkit stable-diffusion ggml
        easy-diffusion-ui cosmo-ggml-webgpu-adapter)
endif()
