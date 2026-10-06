# Keep all vendored vocabulary bytes and loader bodies, but avoid parsing the
# twelve large byte-array headers in one compiler process (over 230 MiB of C++).
set(_vocab_root "${SD_SOURCE}/src/tokenizers/vocab")
file(READ "${_vocab_root}/vocab.cpp" _vocab_source)
set(_vocab_pairs
    "load_clip_merges|clip_merges.hpp"
    "load_qwen2_merges|qwen_merges.hpp"
    "load_mistral_merges|mistral_merges.hpp"
    "load_mistral_vocab_json|mistral_vocab.hpp"
    "load_t5_tokenizer_json|t5.hpp"
    "load_umt5_tokenizer_json|umt5.hpp"
    "load_gemma_merges|gemma_merges.hpp"
    "load_gemma_vocab_json|gemma_vocab.hpp"
    "load_gemma2_merges|gemma2_merges.hpp"
    "load_gemma2_vocab_json|gemma2_vocab.hpp"
    "load_gpt_oss_merges|gpt_oss_merges.hpp"
    "load_gpt_oss_vocab_json|gpt_oss_vocab.hpp")
string(REGEX MATCHALL "std::string load_[a-z0-9_]+\\(\\)" _vocab_functions "${_vocab_source}")
list(LENGTH _vocab_functions _vocab_count)
if(NOT _vocab_count EQUAL 12)
    message(FATAL_ERROR "The pinned vocabulary loader layout changed")
endif()
get_target_property(_sd_sources stable-diffusion SOURCES)
list(LENGTH _sd_sources _sd_source_count)
list(FILTER _sd_sources EXCLUDE REGEX "(^|/)src/tokenizers/vocab/vocab\\.cpp$")
list(LENGTH _sd_sources _sd_without_vocab_count)
math(EXPR _expected_count "${_sd_source_count} - 1")
if(NOT _sd_without_vocab_count EQUAL _expected_count)
    message(FATAL_ERROR "Expected exactly one SD vocabulary translation unit")
endif()
set_property(TARGET stable-diffusion PROPERTY SOURCES "${_sd_sources}")
foreach(_pair IN LISTS _vocab_pairs)
    string(REPLACE "|" ";" _parts "${_pair}")
    list(GET _parts 0 _function)
    list(GET _parts 1 _header)
    string(REGEX MATCH "std::string ${_function}\\(\\) \\{[^}]*\\}" _body "${_vocab_source}")
    if(NOT _body OR NOT EXISTS "${_vocab_root}/${_header}")
        message(FATAL_ERROR "Missing pinned vocabulary function or data: ${_pair}")
    endif()
    set(_output "${CMAKE_BINARY_DIR}/generated/vocab/${_function}.cpp")
    file(GENERATE OUTPUT "${_output}" CONTENT
        "// Generated from the unchanged vendored vocabulary loader.\n#include \"${_vocab_root}/vocab.h\"\n#include \"${_vocab_root}/${_header}\"\n${_body}\n")
    set_source_files_properties("${_output}" TARGET_DIRECTORY stable-diffusion PROPERTIES GENERATED TRUE)
    target_sources(stable-diffusion PRIVATE "${_output}")
endforeach()
