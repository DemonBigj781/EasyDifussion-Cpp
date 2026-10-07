/* SPDX-License-Identifier: MIT
 * Compiled by the HOST compiler. Its TLS belongs to host libc, not Cosmo. */
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
static __thread unsigned long host_tls = 0x123456780000UL;
unsigned long native_tls_probe(void) { return ++host_tls; }
uintptr_t native_self_probe(void) { return (uintptr_t)pthread_self(); }
uintptr_t native_errno_probe(void) { return (uintptr_t)&errno; }
uintptr_t native_fs_probe(void) { uintptr_t value; __asm__("movq %%fs:0,%0":"=r"(value)); return value; }
