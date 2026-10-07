#include "inference_worker.hpp"
#include "logging.h"
#include <pthread.h>
#include <stdexcept>
#include <cstdio>
__attribute__((noinline)) static void stack_probe() {
    volatile unsigned char bytes[256*1024];
    for (unsigned i=0;i<sizeof(bytes);i+=4096) bytes[i]=(unsigned char)(i/4096);
    for (unsigned i=0;i<sizeof(bytes);i+=4096) if(bytes[i]!=(unsigned char)(i/4096)) throw std::runtime_error("stack-data");
}
int main() {
    auto parent=pthread_self();
    auto result=cosmo_inference_worker([&] {
        if(pthread_equal(parent,pthread_self())) throw std::runtime_error("not-worker");
        stack_probe();return std::vector<std::string>{"joined-result"};
    });
    if(result.size()!=1||result[0]!="joined-result")return 1;
    try {cosmo_inference_worker([]()->std::vector<std::string>{throw std::invalid_argument("original-type");});return 2;}
    catch(const std::invalid_argument&e){if(std::string(e.what())!="original-type")return 3;}
    try {cosmo_inference_worker([]()->std::vector<std::string>{sd_log_cb(SD_LOG_ERROR,"Vulkan ErrorOutOfDeviceMemory",nullptr);throw std::runtime_error("allocation-failed");});return 4;}
    catch(const std::runtime_error&e){if(std::string(e.what())!="allocation-failed")return 5;}
    if(!sd_generation_error_out_of_vram()||sd_generation_error_message("fallback").find("Out of VRAM")==std::string::npos)return 6;
    cosmo_inference_worker([] {if(sd_generation_error_out_of_vram())throw std::runtime_error("stale-worker-TLS");return std::vector<std::string>{};});
    if(sd_generation_error_out_of_vram())return 7;
    puts("INFERENCE_WORKER_CHECK return=pass stack_256k=pass exception_type=pass oom_tls=pass tls_reset=pass PASS");return 0;
}
