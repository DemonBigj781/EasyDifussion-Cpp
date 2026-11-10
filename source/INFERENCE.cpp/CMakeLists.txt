cmake_minimum_required(VERSION 3.16)
project(edcpp_inference LANGUAGES CXX)

set(DIFFUSER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../DiffUser.cpp" CACHE PATH
    "Path to the DiffUser.cpp Common source root")

add_library(edcpp_inference STATIC
    src/plan.cpp
    features/conditioning/common/prompt_attention.cpp
    features/conditioning/common/token_sequence.cpp
    features/token/common/tokenizer.cpp
    features/token/common/tokenizer_resource.cpp
    features/token/common/t5_unigram.cpp
    features/token/common/t5_vocabulary.cpp
    Feature/vae/common/vae.cpp
)
target_include_directories(edcpp_inference PUBLIC
    include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DIFFUSER_ROOT}
)
target_compile_features(edcpp_inference PUBLIC cxx_std_17)

include(CTest)
if(BUILD_TESTING)
    add_executable(edcpp_inference_plan_test tests/plan_test.cpp)
    target_link_libraries(edcpp_inference_plan_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_plan COMMAND edcpp_inference_plan_test)

    add_executable(edcpp_inference_tokenizer_test tests/features/token/tokenizer_test.cpp)
    target_link_libraries(edcpp_inference_tokenizer_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_tokenizer COMMAND edcpp_inference_tokenizer_test)

    add_executable(edcpp_inference_prompt_attention_test tests/features/conditioning/prompt_attention_test.cpp)
    target_link_libraries(edcpp_inference_prompt_attention_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_prompt_attention COMMAND edcpp_inference_prompt_attention_test)

    add_executable(edcpp_inference_token_sequence_test tests/features/conditioning/token_sequence_test.cpp)
    target_link_libraries(edcpp_inference_token_sequence_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_token_sequence COMMAND edcpp_inference_token_sequence_test)

    add_executable(edcpp_inference_t5_unigram_test tests/features/token/t5_unigram_test.cpp)
    target_link_libraries(edcpp_inference_t5_unigram_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_t5_unigram COMMAND edcpp_inference_t5_unigram_test)

    add_executable(edcpp_inference_tokenizer_resource_test tests/features/token/tokenizer_resource_test.cpp)
    target_link_libraries(edcpp_inference_tokenizer_resource_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_tokenizer_resource COMMAND edcpp_inference_tokenizer_resource_test)

    add_executable(edcpp_inference_tokenizer_golden_runner tests/features/token/tokenizer_golden_runner.cpp)
    target_link_libraries(edcpp_inference_tokenizer_golden_runner PRIVATE edcpp_inference)
    add_test(
        NAME edcpp_inference_tokenizer_golden_clip_smoke
        COMMAND edcpp_inference_tokenizer_golden_runner
            clip-merges
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/token/clip_tiny.merges
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/token/clip_tiny.tsv
    )
    add_test(
        NAME edcpp_inference_tokenizer_golden_t5_smoke
        COMMAND edcpp_inference_tokenizer_golden_runner
            t5-json
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/token/t5_tiny.json
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/token/t5_tiny.tsv
    )

    add_executable(edcpp_inference_vae_plan_test tests/Feature/vae/vae_plan_test.cpp)
    target_link_libraries(edcpp_inference_vae_plan_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_vae_plan COMMAND edcpp_inference_vae_plan_test)
endif()
