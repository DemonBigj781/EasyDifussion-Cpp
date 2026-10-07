/* Isolated HTTP lifecycle probe. This deliberately substitutes the expensive
   model call and is never linked into the packaged application. */
#include "inference_worker.hpp"
#include <atomic>
#include <cstdio>
#include <unistd.h>

using ProbeWork = std::function<std::vector<std::string>()>;
extern "C" std::vector<std::string> http_probe_work(ProbeWork)
    asm("__wrap__Z22cosmo_inference_workerNSt3__18functionIFNS_6vectorINS_12basic_stringIcNS_11char_traitsIcEENS_9allocatorIcEEEENS5_IS7_EEEEvEEE");

extern "C" std::vector<std::string> http_probe_work(ProbeWork work) {
    (void) work;
    static std::atomic<unsigned> calls{0};
    const unsigned call = ++calls;
    std::fprintf(stderr, "HTTP_PROBE_WORK_STARTED call=%u delay_ms=2000 model_called=0\n", call);
    std::fflush(stderr);
    for (unsigned i = 0; i < 100; ++i) usleep(20000);
    std::fprintf(stderr, "HTTP_PROBE_WORK_FINISHED call=%u\n", call);
    std::fflush(stderr);
    return {"cHJvYmU="};
}
