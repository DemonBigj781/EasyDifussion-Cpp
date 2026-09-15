add_executable(model-loader-version-test model_loader_version_test.cpp)
target_link_libraries(model-loader-version-test PRIVATE stable-diffusion)
add_test(NAME model-loader-version-test COMMAND model-loader-version-test)

add_executable(model-manager-lora-cache-test model_manager_lora_cache_test.cpp)
target_link_libraries(model-manager-lora-cache-test PRIVATE stable-diffusion)
add_test(NAME model-manager-lora-cache-test COMMAND model-manager-lora-cache-test)

add_executable(clip-token-count clip_token_count.cpp)
target_link_libraries(clip-token-count PRIVATE stable-diffusion)
add_test(NAME clip-token-count-test COMMAND clip-token-count --expect 1 hello)

add_executable(clip-vision-architecture-test clip_vision_architecture_test.cpp)
target_link_libraries(clip-vision-architecture-test PRIVATE stable-diffusion)
target_compile_definitions(clip-vision-architecture-test PRIVATE GGML_MAX_NAME=160)
add_test(NAME clip-vision-architecture-test COMMAND clip-vision-architecture-test)

add_executable(backend-assignment-test backend_assignment_test.cpp)
target_link_libraries(backend-assignment-test PRIVATE stable-diffusion)
add_test(NAME backend-assignment-test COMMAND backend-assignment-test)

add_executable(gguf-original-shape-test gguf_original_shape_test.cpp)
target_link_libraries(gguf-original-shape-test PRIVATE stable-diffusion)
add_test(NAME gguf-original-shape-test COMMAND gguf-original-shape-test)

add_executable(sampler-scheduler-test sampler_scheduler_test.cpp)
target_link_libraries(sampler-scheduler-test PRIVATE stable-diffusion)
target_compile_definitions(sampler-scheduler-test PRIVATE GGML_MAX_NAME=160)
add_test(NAME sampler-scheduler-test COMMAND sampler-scheduler-test)

add_executable(backend-device-info-test backend_device_info_test.cpp)
target_link_libraries(backend-device-info-test PRIVATE stable-diffusion)
add_test(NAME backend-device-info-test COMMAND backend-device-info-test)
