# Tools CMakeLists.txt

# Find Threads package for multithreading support (needed by all tools)
find_package(Threads REQUIRED)

# Set target-agnostic include directories for all targets in this directory
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../stable-diffusion.cpp
)

# Set target-agnostic compile options for all targets in this directory
if(MSVC)
    add_compile_options(/wd4566)  # Suppress Unicode character warnings from Crow library headers
endif()

# Function to create a tool with minimal target-specific configuration
function(add_sdkit_tool TOOL_NAME)
    # Parse arguments - first arg is tool name, rest are source files
    set(SOURCE_FILES ${ARGN})

    # Create executable
    add_executable(${TOOL_NAME} ${SOURCE_FILES})

    # Set RPATH for Linux/Unix to find shared libraries
    if(UNIX AND NOT APPLE)
        set_target_properties(${TOOL_NAME} PROPERTIES
            BUILD_RPATH "$ORIGIN:$ORIGIN/../lib"
            INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()

    # For macOS, use @executable_path
    if(APPLE)
        set_target_properties(${TOOL_NAME} PROPERTIES
            BUILD_RPATH "@executable_path:@executable_path/../lib"
            INSTALL_RPATH "@executable_path:@executable_path/../lib"
            INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()

    # Common link libraries
    target_link_libraries(${TOOL_NAME} PRIVATE ggml)
    target_link_libraries(${TOOL_NAME} PRIVATE stable-diffusion)
    target_link_libraries(${TOOL_NAME} PRIVATE Threads::Threads)

    # Platform-specific libraries
    if(WIN32)
        target_link_libraries(${TOOL_NAME} PRIVATE ws2_32)

        # Copy the necessary DLLs to the tools output directory on Windows
        add_custom_command(TARGET ${TOOL_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:stable-diffusion>
                $<TARGET_FILE:ggml>
                $<TARGET_FILE_DIR:${TOOL_NAME}>
        )
    endif()
endfunction()

# Define tools here - just specify the tool name and its source files
add_sdkit_tool(modelinfo
    modelinfo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/model_detection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/logging.cpp
)
