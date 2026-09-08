cmake_minimum_required(VERSION 3.16)
project(edcpp_inference LANGUAGES CXX)

set(DIFFUSER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../DiffUser.cpp" CACHE PATH
    "Path to the DiffUser.cpp Common source root")

add_library(edcpp_inference STATIC
    src/plan.cpp
    features/token/common/tokenizer.cpp
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

    add_executable(edcpp_inference_vae_plan_test tests/Feature/vae/vae_plan_test.cpp)
    target_link_libraries(edcpp_inference_vae_plan_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_vae_plan COMMAND edcpp_inference_vae_plan_test)
endif()
