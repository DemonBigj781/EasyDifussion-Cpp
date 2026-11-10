# Opt-in integration with the caller's already configured GGML CPU runtime.
# Never create another GGML tree or enumerate GPU devices.
if(NOT TARGET ggml OR NOT TARGET ggml-cpu)
    message(FATAL_ERROR "Theory CPU resources require existing ggml and ggml-cpu targets")
endif()
add_library(theory-cpu-resources STATIC
    features/load/common/devices/load.cpp
    features/load/cpu/definition/cpu/model.cpp
    features/load/cpu/translation/cpu/model.cpp
    features/unload/common/devices/unload.cpp
    features/unload/cpu/cpu/definition/model.cpp
    features/unload/cpu/cpu/translation/model.cpp
    native/cpu/byte_storage.cpp
    library/src/cpu_byte_buffer.cpp
    library/src/gguf_cpu_weights.cpp
    features/weights/common/weights.cpp
    features/weights/cpu/translation/weights.cpp
    native/cpu/gguf_weights.cpp)
target_compile_features(theory-cpu-resources PUBLIC cxx_std_17)
target_compile_definitions(theory-cpu-resources PRIVATE EDCPP_USE_GGML_CPU_RESOURCE=1)
target_include_directories(theory-cpu-resources
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/library/include"
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(theory-cpu-resources PRIVATE ggml ggml-cpu Threads::Threads)
add_executable(theory-cpu-resource-test tests/cpu_facade.cpp)
target_compile_definitions(theory-cpu-resource-test PRIVATE EXPECT_GGML_CPU=1)
target_link_libraries(theory-cpu-resource-test PRIVATE theory-cpu-resources)
add_test(NAME theory-cpu-resource COMMAND theory-cpu-resource-test)
foreach(contract cpu_lifecycle cpu_ownership malformed_load)
    add_executable(theory-${contract}-test tests/${contract}.cpp)
    target_include_directories(theory-${contract}-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(theory-${contract}-test PRIVATE theory-cpu-resources)
    add_test(NAME theory-${contract} COMMAND theory-${contract}-test)
endforeach()
add_custom_target(theory-cpu-resource-tests
    DEPENDS theory-cpu-resource-test theory-cpu_lifecycle-test
            theory-cpu_ownership-test theory-malformed_load-test)
add_executable(theory-gguf-weights-test tests/gguf_cpu_weights.cpp)
target_include_directories(theory-gguf-weights-test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(theory-gguf-weights-test PRIVATE theory-cpu-resources ggml ggml-cpu)
add_test(NAME theory-gguf-weights COMMAND theory-gguf-weights-test "${CMAKE_CURRENT_BINARY_DIR}")
add_dependencies(theory-cpu-resource-tests theory-gguf-weights-test)
