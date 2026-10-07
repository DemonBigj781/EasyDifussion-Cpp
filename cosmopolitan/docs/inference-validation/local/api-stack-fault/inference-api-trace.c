#include <cosmo.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>
#include "stable-diffusion.h"

extern bool __real_generate_image(sd_ctx_t *, const sd_img_gen_params_t *, sd_image_t **, int *);
bool __wrap_generate_image(sd_ctx_t *, const sd_img_gen_params_t *, sd_image_t **, int *);

static _Thread_local uintptr_t trace_stack_low;
static _Thread_local uintptr_t trace_stack_high;
static _Thread_local uintptr_t trace_alt_low;
static _Thread_local uintptr_t trace_alt_high;

struct trace_text { char bytes[8192]; size_t used; };

static void trace_literal(struct trace_text *out, const char *text) {
    while (*text && out->used < sizeof(out->bytes)) out->bytes[out->used++] = *text++;
}

static void trace_hex(struct trace_text *out, const char *name, uint64_t value) {
    static const char digits[] = "0123456789abcdef";
    trace_literal(out, name);
    trace_literal(out, "0x");
    for (int shift = 60; shift >= 0; shift -= 4) {
        if (out->used < sizeof(out->bytes)) out->bytes[out->used++] = digits[(value >> shift) & 15];
    }
    trace_literal(out, " ");
}

static void trace_fault(int number, siginfo_t *info, void *opaque) {
    const ucontext_t *context = opaque;
    struct trace_text out = {{0}, 0};
    trace_literal(&out, "API_TRACE_FAULT ");
    trace_hex(&out, "signal=", (unsigned)number);
    trace_hex(&out, "code=", info ? (uint32_t)info->si_code : 0);
    trace_hex(&out, "address=", info ? (uintptr_t)info->si_addr : 0);
    trace_hex(&out, "handler_sp=", (uintptr_t)&out);
    trace_hex(&out, "stack_low=", trace_stack_low);
    trace_hex(&out, "stack_high=", trace_stack_high);
    trace_hex(&out, "alt_low=", trace_alt_low);
    trace_hex(&out, "alt_high=", trace_alt_high);
    if (context) {
        const mcontext_t *m = &context->uc_mcontext;
        trace_hex(&out, "rip=", m->rip);
        trace_hex(&out, "rsp=", m->rsp);
        trace_hex(&out, "rbp=", m->rbp);
        trace_hex(&out, "rax=", m->rax);
        trace_hex(&out, "rbx=", m->rbx);
        trace_hex(&out, "rcx=", m->rcx);
        trace_hex(&out, "rdx=", m->rdx);
        trace_hex(&out, "rdi=", m->rdi);
        trace_hex(&out, "rsi=", m->rsi);
        trace_hex(&out, "r8=", m->r8);
        trace_hex(&out, "r9=", m->r9);
        trace_hex(&out, "r10=", m->r10);
        trace_hex(&out, "r11=", m->r11);
        trace_hex(&out, "r12=", m->r12);
        trace_hex(&out, "r13=", m->r13);
        trace_hex(&out, "r14=", m->r14);
        trace_hex(&out, "r15=", m->r15);
        trace_literal(&out, "\n");
        uintptr_t frame = m->rbp;
        for (unsigned depth = 0; depth < 32; ++depth) {
            if (!trace_stack_low || frame < trace_stack_low || trace_stack_high < 16 ||
                frame > trace_stack_high - 16 || (frame & 7)) break;
            const volatile uintptr_t *words = (const volatile uintptr_t *)frame;
            uintptr_t next = words[0];
            uintptr_t caller = words[1];
            trace_literal(&out, "API_TRACE_FRAME ");
            trace_hex(&out, "depth=", depth);
            trace_hex(&out, "frame=", frame);
            trace_hex(&out, "return=", caller);
            trace_literal(&out, "\n");
            if (next <= frame) break;
            frame = next;
        }
    }
    trace_literal(&out, "\n");
    size_t written = 0;
    while (written < out.used) {
        ssize_t count = write(STDERR_FILENO, out.bytes + written, out.used - written);
        if (count > 0) written += (size_t)count;
        else if (count < 0 && errno == EINTR) continue;
        else break;
    }
    _Exit(128 + number);
}

bool __wrap_generate_image(sd_ctx_t *context, const sd_img_gen_params_t *params,
                           sd_image_t **images, int *count) {
    const char *enabled = getenv("COSMO_API_TRACE_FAULT");
    if (!enabled || strcmp(enabled, "1")) return __real_generate_image(context, params, images, count);

    pthread_attr_t attr;
    void *stack_base = NULL;
    size_t stack_size = 0, guard_size = 0;
    int attr_status = pthread_getattr_np(pthread_self(), &attr);
    int stack_status = attr_status;
    if (!attr_status) {
        stack_status = pthread_attr_getstack(&attr, &stack_base, &stack_size);
        (void)pthread_attr_getguardsize(&attr, &guard_size);
        pthread_attr_destroy(&attr);
    }
    if (!stack_status) {
        trace_stack_low = (uintptr_t)stack_base;
        trace_stack_high = trace_stack_low + stack_size;
    }
    fprintf(stderr, "API_TRACE_ENTER context=%p thread=%p stack_base=%p stack_size=%zu guard_size=%zu attr_status=%d stack_status=%d local=%p\n",
            (void *)context, (void *)(uintptr_t)pthread_self(), stack_base, stack_size, guard_size,
            attr_status, stack_status, (void *)&attr);
    fflush(stderr);

    void *alt_memory = malloc(65536);
    if (!alt_memory) { fputs("API_TRACE_SETUP_FAIL malloc\n", stderr); return false; }
    stack_t alternate = {.ss_sp=alt_memory, .ss_flags=0, .ss_size=65536}, previous_stack;
    if (sigaltstack(&alternate, &previous_stack)) {
        fprintf(stderr, "API_TRACE_SETUP_FAIL sigaltstack errno=%d\n", errno);
        free(alt_memory);
        return false;
    }
    trace_alt_low = (uintptr_t)alt_memory;
    trace_alt_high = trace_alt_low + 65536;
    struct sigaction action, previous_action;
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = trace_fault;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGSEGV, &action, &previous_action)) {
        fprintf(stderr, "API_TRACE_SETUP_FAIL sigaction errno=%d\n", errno);
        if (!sigaltstack(&previous_stack, NULL)) free(alt_memory);
        return false;
    }
    fprintf(stderr, "API_TRACE_ARMED alt_base=%p alt_size=65536 handler=%p\n", alt_memory, (void *)(uintptr_t)trace_fault);
    fflush(stderr);
    bool result = __real_generate_image(context, params, images, count);
    int restore_signal = sigaction(SIGSEGV, &previous_action, NULL);
    int restore_stack = sigaltstack(&previous_stack, NULL);
    if (!restore_stack) free(alt_memory);
    fprintf(stderr, "API_TRACE_RETURN result=%d restore_signal=%d restore_stack=%d\n", result, restore_signal, restore_stack);
    fflush(stderr);
    return result;
}
