/* SPDX-License-Identifier: MIT */
#ifndef COSMO_INFERENCE_WORKER_HPP
#define COSMO_INFERENCE_WORKER_HPP

#include <functional>
#include <string>
#include <vector>

/* Synchronous request boundary: references captured by work remain alive until
   its explicitly sized worker is joined. Native cancellation uses the existing
   ImageGenerator context and never cancels this pthread. */
std::vector<std::string> cosmo_inference_worker(
    std::function<std::vector<std::string>()> work);

#endif
