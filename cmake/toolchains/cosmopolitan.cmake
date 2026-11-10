# Opt-in, multi-OS/multi-architecture Cosmopolitan toolchain.
# Use a fresh build tree. This is not a toolchain for GPU backends or SDKIT3.
set(EDCPP_COSMOCC_ROOT "" CACHE PATH "Extracted Cosmopolitan cosmocc toolchain")
if(NOT EDCPP_COSMOCC_ROOT)
    message(FATAL_ERROR
        "Set EDCPP_COSMOCC_ROOT to an extracted cosmocc release directory. "
        "See docs/cosmopolitan-compiler-migration.md. No tools are downloaded automatically.")
endif()
get_filename_component(EDCPP_COSMOCC_ROOT "${EDCPP_COSMOCC_ROOT}" ABSOLUTE)
foreach(_cosmo_tool cosmocc cosmoc++ cosmoar)
    if(NOT EXISTS "${EDCPP_COSMOCC_ROOT}/bin/${_cosmo_tool}")
        message(FATAL_ERROR "Missing Cosmopolitan tool: ${EDCPP_COSMOCC_ROOT}/bin/${_cosmo_tool}")
    endif()
endforeach()

# Do not describe a fat executable as a native Linux/Windows target. Generic
# prevents host-specific shared-library and SDK discovery rules being selected.
# OS decisions in portable source must be runtime decisions, not WIN32/APPLE.
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR ape)
set(EDCPP_COSMOPOLITAN TRUE)
set(CMAKE_USER_MAKE_RULES_OVERRIDE "${CMAKE_CURRENT_LIST_DIR}/cosmopolitan-rules.cmake")
foreach(_cosmo_language C CXX)
    if(_cosmo_language STREQUAL "C")
        set(_cosmo_driver cosmocc)
    else()
        set(_cosmo_driver cosmoc++)
    endif()
    set(_cosmo_compiler "${EDCPP_COSMOCC_ROOT}/bin/${_cosmo_driver}")
    if(DEFINED CMAKE_${_cosmo_language}_COMPILER
       AND NOT "${CMAKE_${_cosmo_language}_COMPILER}" STREQUAL "${_cosmo_compiler}")
        message(FATAL_ERROR
            "Conflicting ${_cosmo_language} compiler. Select a fresh Cosmopolitan build directory; "
            "do not reuse a native/GPU CMake cache.")
    endif()
    set(CMAKE_${_cosmo_language}_COMPILER "${_cosmo_compiler}")
    set(CMAKE_${_cosmo_language}_COMPILER_AR "${EDCPP_COSMOCC_ROOT}/bin/cosmoar")

    # cosmoar manages both the visible x86-64 archive and its hidden .aarch64
    # companion. Index both with `cosmoar s`, not a host-only ranlib invocation.
    set(CMAKE_${_cosmo_language}_ARCHIVE_CREATE "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_${_cosmo_language}_ARCHIVE_APPEND "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_${_cosmo_language}_ARCHIVE_FINISH "<CMAKE_AR> s <TARGET>")
endforeach()
set(CMAKE_AR "${EDCPP_COSMOCC_ROOT}/bin/cosmoar")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES EDCPP_COSMOCC_ROOT)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Portable targets use static libraries")
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_SKIP_RPATH TRUE)

# Host build tools remain available, but target headers/libraries/packages must
# come from the Cosmopolitan prefix, never the host's glibc/SDK installation.
list(APPEND CMAKE_FIND_ROOT_PATH "${EDCPP_COSMOCC_ROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
