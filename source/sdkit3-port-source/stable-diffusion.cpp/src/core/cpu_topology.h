#ifndef __SD_CORE_CPU_TOPOLOGY_H__
#define __SD_CORE_CPU_TOPOLOGY_H__

#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_set>

static inline int32_t count_sysfs_physical_cores(const std::string& cpu_root) {
    std::unordered_set<std::string> siblings;
    for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu) {
        std::ifstream thread_siblings(cpu_root + "/cpu" + std::to_string(cpu) + "/topology/thread_siblings");
        if (!thread_siblings.is_open()) break;
        std::string line;
        if (std::getline(thread_siblings, line)) siblings.insert(line);
    }
    return static_cast<int32_t>(siblings.size());
}

#endif  // __SD_CORE_CPU_TOPOLOGY_H__
