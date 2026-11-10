# Register app-owned bridges on the runtime itself. Both native SDKit and the
# Cosmopolitan build add stable-diffusion directly and must use this same list.
function(sdkit_add_runtime_adapters runtime_target sdkit_root)
    if(NOT TARGET "${runtime_target}")
        message(FATAL_ERROR "SDKit runtime adapter target does not exist: ${runtime_target}")
    endif()
    target_sources("${runtime_target}" PRIVATE "${sdkit_root}/runtime_adapters/prompt_validation.cpp")
    target_include_directories("${runtime_target}" PRIVATE "${sdkit_root}/include")
endfunction()
