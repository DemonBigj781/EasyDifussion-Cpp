cmake_minimum_required(VERSION 3.16)
project(edcpp_inference LANGUAGES CXX)

add_library(edcpp_inference STATIC src/plan.cpp)
target_include_directories(edcpp_inference PUBLIC include)
target_compile_features(edcpp_inference PUBLIC cxx_std_17)

include(CTest)
if(BUILD_TESTING)
    add_executable(edcpp_inference_plan_test tests/plan_test.cpp)
    target_link_libraries(edcpp_inference_plan_test PRIVATE edcpp_inference)
    add_test(NAME edcpp_inference_plan COMMAND edcpp_inference_plan_test)
endif()
