cmake_minimum_required(VERSION 3.16)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(easy-diffusion-ui LANGUAGES CXX)
endif()

add_library(easy-diffusion-ui STATIC
    src/ui.cpp
)

target_include_directories(easy-diffusion-ui
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(easy-diffusion-ui PUBLIC cxx_std_17)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../Pages ${CMAKE_CURRENT_BINARY_DIR}/pages)

add_executable(easy-diffusion-ui-render ../Pages/src/render_main.cpp)
target_link_libraries(easy-diffusion-ui-render PRIVATE easy-diffusion-ui)
set_target_properties(easy-diffusion-ui-render PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
