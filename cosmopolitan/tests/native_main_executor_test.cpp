/* SPDX-License-Identifier: MIT
 * Exercises the actual main executor and inference worker with a host TLS
 * fixture and host Vulkan loader. The Vulkan check enumerates instance
 * extensions only; it does not establish a device or shader-dispatch result. */
#include "native_main_executor.hpp"
#include "inference_worker.hpp"
#include <cosmo.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
static thread_local bool oom;
void reset_sd_generation_error() { oom=false; }
bool sd_generation_error_out_of_vram() { return oom; }
void set_sd_generation_error_out_of_vram(bool x) { oom=x; }
extern "C" const char *cosmo_webgpu_requested_provider() { return "native"; }
using TLS=unsigned long(*)(void);
using GetProc=void*(*)(uintptr_t,const char*);
using Extensions=int(*)(const char*,uint32_t*,void*);
int main(int argc,char**argv) {
    if(argc!=2)return 2;
    void *lib=cosmo_dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
    if(!lib){std::fprintf(stderr,"host library: %s\n",cosmo_dlerror());return 3;}
    TLS tls=(TLS)cosmo_dltramp(cosmo_dlsym(lib,"native_tls_probe"));
    void *vk=cosmo_dlopen("libvulkan.so.1",RTLD_NOW|RTLD_LOCAL);
    if(!vk)return 4;
    auto proc=(GetProc)cosmo_dltramp(cosmo_dlsym(vk,"vkGetInstanceProcAddr"));
    auto extensions=(Extensions)cosmo_dltramp(proc(0,"vkEnumerateInstanceExtensionProperties"));
    const auto main_id=pthread_self();unsigned checks=0;
    auto check=[&](bool ok){if(!ok)throw std::runtime_error("lane check failed");++checks;};
    check(tls()==0x123456780001UL);
    cosmo_native_main_service([&] {
        check(!pthread_equal(pthread_self(),main_id));
        auto result=cosmo_inference_worker([&] {
            check(pthread_equal(pthread_self(),main_id));
            check(tls()==0x123456780002UL);
            uint32_t n=0;check(extensions(0,&n,nullptr)==0);
            std::printf("NATIVE_LANE_VULKAN extensions=%u tid=%d\n",n,gettid());
            set_sd_generation_error_out_of_vram(true);
            return std::vector<std::string>{"native-result"};
        });
        check(result.size()==1&&result[0]=="native-result");
        check(sd_generation_error_out_of_vram());
        bool caught=false;
        try {
            cosmo_inference_worker([&]() -> std::vector<std::string> {
                check(pthread_equal(pthread_self(),main_id));
                check(tls()==0x123456780003UL);
                throw std::logic_error("original-type-and-message");
            });
        } catch(const std::logic_error &e) { caught=std::strcmp(e.what(),"original-type-and-message")==0; }
        check(caught);check(!sd_generation_error_out_of_vram());
    });
    bool closed=false;
    try {cosmo_native_main_invoke([]{});}catch(const std::runtime_error&){closed=true;}
    check(closed);
    bool propagated=false;
    try {cosmo_native_main_service([]{throw std::logic_error("service-stopped");});}
    catch(const std::logic_error&e){propagated=std::strcmp(e.what(),"service-stopped")==0;}
    check(propagated);
    check(tls()==0x123456780004UL);
    std::printf("NATIVE_MAIN_EXECUTOR checks=%u PASS\n",checks);
}
