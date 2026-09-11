add_executable(model-loader-version-test model_loader_version_test.cpp)
target_link_libraries(model-loader-version-test PRIVATE stable-diffusion)
add_test(NAME model-loader-version-test COMMAND model-loader-version-test)

add_executable(clip-token-count clip_token_count.cpp)
target_link_libraries(clip-token-count PRIVATE stable-diffusion)
add_test(NAME clip-token-count-test COMMAND clip-token-count --expect 1 hello)

add_executable(gguf-original-shape-test gguf_original_shape_test.cpp)
target_link_libraries(gguf-original-shape-test PRIVATE stable-diffusion)
add_test(NAME gguf-original-shape-test COMMAND gguf-original-shape-test)
