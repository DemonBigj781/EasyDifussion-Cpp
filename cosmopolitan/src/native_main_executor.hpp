/* SPDX-License-Identifier: MIT */
#ifndef COSMO_NATIVE_MAIN_EXECUTOR_HPP
#define COSMO_NATIVE_MAIN_EXECUTOR_HPP
#include <functional>

/* Linux's host libc TLS belongs to the original thread that initialized
   cosmo_dlopen. Keep native Vulkan calls there; never borrow its FS on a
   Cosmopolitan pthread. Embedded-only execution keeps the ordinary workers. */
bool cosmo_native_main_required();

/* Called on the original main thread after registry/device initialization.
   Runs the HTTP service on an explicitly sized thread and pumps synchronous
   native work until that service has stopped and joined its coordinators.
   The caller constructs and destroys the server on the main thread. */
void cosmo_native_main_service(std::function<void()> service);

/* Synchronous: captured objects remain alive until completion; exceptions
   propagate to the submitting coordinator. Rejects calls without a live lane. */
void cosmo_native_main_invoke(std::function<void()> work);
#endif
