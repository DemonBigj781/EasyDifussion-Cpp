#include "platform.hpp"
#include <cstdlib>
#include <stdexcept>
#include <unistd.h>
#if defined(__COSMOPOLITAN__)
#include <cosmo.h>
#if defined(__x86_64__)
#include <libc/nt/dll.h>
#include <libc/nt/runtime.h>
#endif
#endif
#if defined(__x86_64__)
#include <cpuid.h>
#endif

bool on_windows() {
#ifdef __COSMOPOLITAN__
    return IsWindows();
#else
    return false;
#endif
}

Host detect_host() {
    Host h;
#ifdef __COSMOPOLITAN__
    if (IsWindows()) h.os = "win";
    else if (IsLinux()) h.os = "linux";
    else if (IsXnu()) h.os = "mac";
    else if (IsFreebsd()) h.os = "freebsd";
    else if (IsOpenbsd()) h.os = "openbsd";
    else if (IsNetbsd()) h.os = "netbsd";
#elif defined(__linux__)
    h.os = "linux";
#elif defined(__APPLE__)
    h.os = "mac";
#endif
    // Use the executing ABI, including when running under emulation. Windows
    // ARM is not blacklisted; a compatible x64 process may select x64 workers.
#if defined(__aarch64__)
    h.arch = "arm64";
    h.features.insert("neon");
#elif defined(__x86_64__)
    h.arch = "x64";
    unsigned a, b, c, d;
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        if (c & (1u << 20)) h.features.insert("sse4.2");
        bool avx = false;
        if ((c & (1u << 27)) && (c & (1u << 28))) {
            unsigned lo, hi;
            __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
            avx = (lo & 6) == 6;
        }
        if (avx) h.features.insert("avx");
        if (avx && (c & (1u << 12))) h.features.insert("fma");
        if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
            if (avx && (b & (1u << 5))) h.features.insert("avx2");
            if (b & (1u << 8)) h.features.insert("bmi2");
        }
    }
#endif
    if (h.os.empty() || h.arch.empty()) throw std::runtime_error("Unrecognized host OS or process ABI");
    return h;
}

std::string canonical_path(const std::string& path) {
    char* p = realpath(path.c_str(), nullptr);
    if (!p) throw std::runtime_error("Cannot resolve path: " + path);
    std::string result(p);
    std::free(p);
    return result;
}

std::string executable_path(const char* argv0) {
#ifdef __COSMOPOLITAN__
    if (const char* p = GetProgramExecutableName()) return canonical_path(p);
#elif defined(__linux__)
    char buf[8192];
    auto n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0 && n < static_cast<ssize_t>(sizeof(buf) - 1)) {
        buf[n] = 0;
        return canonical_path(buf);
    }
#endif
    if (std::string(argv0).find('/') != std::string::npos) return canonical_path(argv0);
    throw std::runtime_error("Cannot locate launcher; pass --root explicitly");
}

std::string native_path(std::string path) {
    if (on_windows() && path.size() > 2 && path[0] == '/' && path[2] == '/' &&
        ((path[1] >= 'a' && path[1] <= 'z') || (path[1] >= 'A' && path[1] <= 'Z'))) {
        path = path.substr(1, 1) + ":" + path.substr(2);
    }
    return path;
}

void contain_windows_children() {
#if defined(__COSMOPOLITAN__) && defined(__x86_64__)
    if (!on_windows()) return;
    // Join before spawning: assigning already-running children has a race.
    // This non-inherited handle intentionally lives until process exit. Its
    // final close kills remaining descendants without a global process scan.
    using CreateJob = int64_t(__attribute__((ms_abi)) *)(void*, const char16_t*);
    using SetJob = int32_t(__attribute__((ms_abi)) *)(int64_t, int, void*, uint32_t);
    using AssignJob = int32_t(__attribute__((ms_abi)) *)(int64_t, int64_t);
    auto kernel = GetModuleHandle("kernel32.dll");
    auto create = reinterpret_cast<CreateJob>(GetProcAddress(kernel, "CreateJobObjectW"));
    auto set = reinterpret_cast<SetJob>(GetProcAddress(kernel, "SetInformationJobObject"));
    auto assign = reinterpret_cast<AssignJob>(GetProcAddress(kernel, "AssignProcessToJobObject"));
    struct Limits {
        int64_t process_time, job_time;
        uint32_t flags, pad1;
        uint64_t min_working_set, max_working_set;
        uint32_t active_processes, pad2;
        uint64_t affinity;
        uint32_t priority, scheduling;
        uint64_t io_counters[6];
        uint64_t process_memory, job_memory, peak_process_memory, peak_job_memory;
    } limits{};
    static_assert(sizeof(Limits) == 144, "Windows x64 job limit ABI");
    limits.flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (!create || !set || !assign) throw std::runtime_error("Windows Job Object APIs unavailable");
    auto job = create(nullptr, nullptr);
    if (!job) throw std::runtime_error("Cannot create Windows process containment job");
    if (!set(job, 9, &limits, sizeof(limits)) || !assign(job, GetCurrentProcess())) {
        CloseHandle(job);
        throw std::runtime_error("Cannot contain Windows children in a job; no processes launched");
    }
#endif
}
