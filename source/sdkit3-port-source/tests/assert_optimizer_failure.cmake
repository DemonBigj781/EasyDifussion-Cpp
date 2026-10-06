if(NOT DEFINED TEST_EXE)
    message(FATAL_ERROR "TEST_EXE is required")
endif()

execute_process(COMMAND "${TEST_EXE}" RESULT_VARIABLE normal_result
    OUTPUT_VARIABLE normal_output ERROR_VARIABLE normal_error TIMEOUT 30)
string(STRIP "${normal_output}" normal_output)
if(NOT normal_result STREQUAL "0" OR NOT normal_output STREQUAL "SUCCESS")
    message(FATAL_ERROR "Normal optimizer control failed: ${normal_result}\n${normal_error}")
endif()

execute_process(COMMAND "${TEST_EXE}" abort RESULT_VARIABLE aborted_result
    OUTPUT_VARIABLE aborted_output ERROR_VARIABLE aborted_error TIMEOUT 30)
if(aborted_result STREQUAL "0" OR NOT aborted_error MATCHES "ggml_opt: graph computation failed")
    message(FATAL_ERROR "Aborted graph was not reported correctly: ${aborted_result}\n${aborted_output}\n${aborted_error}")
endif()
message(STATUS "PASS: optimizer rejects an aborted graph; normal execution succeeds")
