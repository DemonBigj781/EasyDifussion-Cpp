/* SPDX-License-Identifier: MIT */
#ifndef COSMO_INFERENCE_WORKER_HPP
#define COSMO_INFERENCE_WORKER_HPP

#include <functional>
#include <string>
#include <vector>

/* Synchronous request boundary: references captured by work remain alive until
   completion. Embedded/Windows use an explicitly sized worker. Linux native
   work uses the original-main executor, preserving host-library TLS. Native
   cancellation uses ImageGenerator's context, never pthread cancellation. */
std::vector<std::string> cosmo_inference_worker(
    std::function<std::vector<std::string>()> work);

#endif
