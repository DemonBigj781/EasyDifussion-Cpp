#include "runtime.h"
#include "llama_service.h"
#ifndef COSMO_LLAMA_REFERENCE_BUILD
#include "config.h"
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"
#ifdef COSMO_WEBGPU_BACKEND
#include "cosmo-webgpu.h"
#endif

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define LLAMA_DEMO_MODEL "/zip/models/stories260K.gguf"
#define LLAMA_DEMO_PROMPT "Once upon a time"
#define LLAMA_SELFTEST_STEPS 16
#define LLAMA_MAX_PROMPT_BYTES (1024u * 1024u)
#define LLAMA_MAX_CONTEXT 32768u
#define LLAMA_MAX_GENERATED 4096
#define LLAMA_MAX_THREADS 256
#define LLAMA_MAX_OUTPUT_BYTES (16u * 1024u * 1024u)

/* Native reference: original llama.cpp c589f0ed10c643678c4707dd160c21ac7633ebc0,
   original GGML, GCC 13.3 baseline CPU, one thread. Model SHA256:
   047bf46455a544931cff6fef14d7910154c56afbc23ab1c5e56a72e69912c04b. */
static const llama_token llama_expected_ids[LLAMA_SELFTEST_STEPS] = {
    432, 383, 286, 261, 376, 298, 315, 421,
    395, 317, 426, 338, 401, 396, 267, 337
};

struct llama_run_options {
    const char *model_path;
    const char *prompt;
    int tokens;
    int threads;
    bool selftest;
    bool webgpu;
    const char *device;
    bool report_tokens;
    const struct cosmo_llama_request *service;
    struct cosmo_llama_result *output;
};

struct llama_recovery_state {
    struct llama_model *model;
    struct llama_context *context;
    llama_token *prompt_tokens;
    char *token_piece;
};

void cosmo_llama_service_cleanup(struct cosmo_llama_result *output) {
    if (output == NULL || output->_state == NULL) return;
    struct llama_recovery_state *state = output->_state;
    if (state->context != NULL) {
        struct llama_context *context = state->context;
        state->context = NULL;
        llama_free(context);
    }
    if (state->model != NULL) {
        struct llama_model *model = state->model;
        state->model = NULL;
        llama_model_free(model);
    }
    free(state->prompt_tokens);
    free(state->token_piece);
    free(state);
    output->_state = NULL;
}

static void llama_error(const struct llama_run_options *options, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    if (options->output != NULL) {
        va_start(args, format);
        vsnprintf(options->output->error, sizeof(options->output->error), format, args);
        va_end(args);
        size_t n = strlen(options->output->error);
        if (n && options->output->error[n - 1] == '\n') options->output->error[n - 1] = '\0';
    }
}

static bool llama_cancelled(const struct llama_run_options *options) {
    return options->service && options->service->cancelled &&
           options->service->cancelled(options->service->user);
}

static bool llama_service_load_progress(float progress, void *opaque) {
    (void)progress;
    return !llama_cancelled(opaque);
}

static bool llama_service_abort(void *opaque) {
    return llama_cancelled(opaque);
}

static void llama_usage(void) {
    fputs("Usage: llama [--model FILE.gguf] [--prompt TEXT] "
          "[--tokens N] [--threads N] [--backend cpu|webgpu] [--device NAME] [--report-tokens]\n"
          "  Default model: embedded trained stories260K.gguf\n"
          "  Default prompt: Once upon a time\n"
          "  Default generation: 64 greedy tokens, 1 CPU thread\n"
          "  --device accepts auto, a listed selector, or an available stable ID.\n"
          "  WebGPU uses the configured native/embedded Vulkan provider.\n"
          "  Limits: 1..4096 tokens, 1..256 threads, 32768 context tokens\n"
          "  This entry point supports decoder-only text models.\n",
          stdout);
}

static int llama_parse_positive(const char *text, int maximum, int *value) {
    char *end;
    long parsed;
    if (text == NULL || text[0] < '0' || text[0] > '9') {
        return -1;
    }
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed < 1 || parsed > maximum) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static void llama_free_piece(const struct llama_run_options *options, char *piece) {
    if (options->output && options->output->_state) {
        struct llama_recovery_state *state = options->output->_state;
        if (state->token_piece == piece) state->token_piece = NULL;
    }
    free(piece);
}

static int llama_write_piece(const struct llama_vocab *vocab,
                             llama_token token, const struct llama_run_options *options) {
    char local[256];
    char *piece = local;
    int32_t length = llama_token_to_piece(vocab, token, local,
                                          (int32_t)sizeof(local), 0, false);
    if (length < 0) {
        if (length == INT32_MIN || -length > (int32_t)LLAMA_MAX_PROMPT_BYTES) {
            llama_error(options, "llama: token text exceeds the output buffer limit\n");
            return -1;
        }
        const int32_t capacity = -length;
        piece = malloc((size_t)capacity);
        if (piece == NULL) {
            llama_error(options, "llama: could not allocate token text\n");
            return -1;
        }
        if (options->output) {
            struct llama_recovery_state *state = options->output->_state;
            state->token_piece = piece;
        }
        length = llama_token_to_piece(vocab, token, piece, capacity, 0, false);
        if (length < 0 || length > capacity) {
            llama_free_piece(options, piece);
            llama_error(options, "llama: failed to render token text\n");
            return -1;
        }
    }
    bool ok = true;
    if (options->output) {
        struct cosmo_llama_result *out = options->output;
        if ((size_t)length > LLAMA_MAX_OUTPUT_BYTES - out->text_bytes) {
            if (piece != local) llama_free_piece(options, piece);
            llama_error(options, "llama: generated text exceeds the 16 MiB output limit\n");
            return -1;
        }
        char *expanded = realloc(out->text, out->text_bytes + (size_t)length + 1);
        if (expanded == NULL) {
            if (piece != local) llama_free_piece(options, piece);
            llama_error(options, "llama: could not allocate generated text\n");
            return -1;
        }
        out->text = expanded;
        memcpy(out->text + out->text_bytes, piece, (size_t)length);
        out->text_bytes += (size_t)length;
        out->text[out->text_bytes] = '\0';
    } else {
        ok = fwrite(piece, 1, (size_t)length, stdout) == (size_t)length;
    }
    if (piece != local) {
        llama_free_piece(options, piece);
    }
    if (!ok || (!options->output && fflush(stdout) != 0)) {
        llama_error(options, "llama: failed to write generated text\n");
        return -1;
    }
    return 0;
}

static int llama_greedy_checked(struct llama_context *context,
                                int32_t vocabulary_size,
                                llama_token *selected, const struct llama_run_options *options) {
    const float *logits = llama_get_logits_ith(context, -1);
    if (logits == NULL || vocabulary_size <= 0) {
        llama_error(options, "llama: decoder produced no logits\n");
        return -1;
    }
    llama_token best = 0;
    for (int32_t i = 0; i < vocabulary_size; ++i) {
        if (!isfinite(logits[i])) {
            llama_error(options, "llama: non-finite logit at token %" PRId32 "\n", i);
            return -1;
        }
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    *selected = best;
    return 0;
}

static int llama_run(const struct llama_run_options *options) {
    int result = 1;
    struct llama_recovery_state *recovery = NULL;
    bool backend_initialized = false;
    struct llama_model *model = NULL;
    struct llama_context *context = NULL;
    llama_token *prompt_tokens = NULL;
    llama_token selftest_ids[LLAMA_MAX_GENERATED];
    int generated = 0;
    int decode_steps = 0;
    int32_t prompt_count = 0;
    const struct llama_vocab *vocab = NULL;
    ggml_backend_dev_t selected_devices[] = {NULL, NULL};
    const char *requested_device = options->device ? options->device : "auto";
#ifdef COSMO_WEBGPU_BACKEND
    const struct cosmo_webgpu_device_info *selected_info = NULL;
#endif
#ifdef COSMO_WEBGPU_BACKEND
    uint64_t graphs_before = 0, dispatches_before = 0, matmuls_before = 0;
    uint64_t submissions_before = 0, readbacks_before = 0;
    uint64_t decode_graphs_before = 0, decode_dispatches_before = 0, decode_matmuls_before = 0;
    uint64_t decode_submissions_before = 0, decode_readbacks_before = 0;
    if (cosmo_webgpu_initialize()) {
        llama_error(options, "llama: WebGPU provider initialization failed\n");
        return 1;
    }
#else
    if (options->webgpu) {
        llama_error(options, "llama: WebGPU was not linked into this build\n");
        return 1;
    }
#endif
    const size_t prompt_bytes = strnlen(options->prompt,
                                       (size_t)LLAMA_MAX_PROMPT_BYTES + 1);

    if (prompt_bytes > LLAMA_MAX_PROMPT_BYTES || prompt_bytes > INT32_MAX) {
        llama_error(options, "llama: prompt exceeds the 1 MiB input limit\n");
        return 2;
    }

    if (options->output) {
        recovery = calloc(1, sizeof(*recovery));
        if (!recovery) {
            llama_error(options, "llama: could not allocate request recovery state\n");
            return 1;
        }
        options->output->_state = recovery;
    }
    if (llama_cancelled(options)) { result = COSMO_LLAMA_CANCELLED; goto cleanup; }

    /* All backend registrations are static; no plugin directory is scanned. */
    if (ggml_backend_reg_by_name("CPU") == NULL) {
        ggml_backend_reg_t cpu = ggml_backend_cpu_reg();
        if (cpu == NULL) {
            llama_error(options, "llama: linked CPU backend is unavailable\n");
            goto cleanup;
        }
        ggml_backend_register(cpu);
    }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) == NULL) {
        llama_error(options, "llama: no registered CPU device\n");
        goto cleanup;
    }
    if (options->webgpu) {
        ggml_backend_reg_t webgpu = ggml_backend_reg_by_name("WebGPU");
        if (webgpu == NULL || ggml_backend_reg_dev_count(webgpu) == 0) {
            llama_error(options, "llama: requested WebGPU backend is unavailable\n");
            goto cleanup;
        }
#ifdef COSMO_WEBGPU_BACKEND
        if (cosmo_webgpu_select_device(requested_device)) {
            llama_error(options, "llama: requested WebGPU device is unavailable: %s\n", requested_device);
            goto cleanup;
        }
        selected_info = cosmo_webgpu_device_metadata("auto");
        if (selected_info == NULL) goto cleanup;
        selected_devices[0] = ggml_backend_dev_by_name(selected_info->selector);
        if (selected_devices[0] == NULL || ggml_backend_dev_backend_reg(selected_devices[0]) != webgpu) {
            llama_error(options, "llama: selected adapter has no matching GGML device\n");
            goto cleanup;
        }
        fprintf(stderr, "LLAMA_DEVICE selector=%s provider=%s software=%d stable_id=%s\n",
                selected_info->selector, selected_info->provider, selected_info->software,
                selected_info->stable_id && *selected_info->stable_id ? selected_info->stable_id : "unavailable");
#endif
    } else if (strcmp(requested_device, "auto") && strcmp(requested_device, "CPU") &&
               strcmp(requested_device, "cpu")) {
        llama_error(options, "llama: device %s does not belong to the CPU backend\n", requested_device);
        result = 2;
        goto cleanup;
    }
    llama_backend_init();
    backend_initialized = true;

    struct llama_model_params model_params = llama_model_default_params();
    model_params.devices = selected_devices;
    model_params.n_gpu_layers = options->webgpu ? -1 : 0;
    model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    model_params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    model_params.check_tensors = true;
    model_params.use_extra_bufts = false;
    model_params.progress_callback = options->service ? llama_service_load_progress : NULL;
    model_params.progress_callback_user_data = (void *)options;
    model = llama_model_load_from_file(options->model_path, model_params);
    if (recovery) recovery->model = model;
    if (llama_cancelled(options)) { result = COSMO_LLAMA_CANCELLED; goto cleanup; }
    if (model == NULL) {
        llama_error(options, "llama: could not load model: %s\n", options->model_path);
        goto cleanup;
    }
    if (llama_model_has_encoder(model) || !llama_model_has_decoder(model)) {
        llama_error(options, "llama: unsupported model: this entry point requires a "
              "decoder-only text model\n");
        goto cleanup;
    }
    vocab = llama_model_get_vocab(model);
    if (vocab == NULL || llama_vocab_n_tokens(vocab) <= 0) {
        llama_error(options, "llama: model has no usable vocabulary\n");
        goto cleanup;
    }

    const int32_t needed = llama_tokenize(vocab, options->prompt,
                                          (int32_t)prompt_bytes,
                                          NULL, 0, true, false);
    if (needed >= 0 || needed == INT32_MIN ||
        (uint32_t)(-needed) > LLAMA_MAX_CONTEXT ||
        (size_t)(-needed) > SIZE_MAX / sizeof(*prompt_tokens)) {
        llama_error(options, "llama: prompt tokenization is empty, invalid, or too large\n");
        goto cleanup;
    }
    prompt_count = -needed;
    if (options->output) options->output->prompt_tokens = prompt_count;
    if ((uint32_t)prompt_count > LLAMA_MAX_CONTEXT - (uint32_t)options->tokens) {
        llama_error(options, "llama: prompt plus generation exceeds 32768 context tokens\n");
        goto cleanup;
    }
    prompt_tokens = malloc((size_t)prompt_count * sizeof(*prompt_tokens));
    if (recovery) recovery->prompt_tokens = prompt_tokens;
    if (prompt_tokens == NULL) {
        llama_error(options, "llama: could not allocate prompt tokens\n");
        goto cleanup;
    }
    if (llama_tokenize(vocab, options->prompt, (int32_t)prompt_bytes,
                       prompt_tokens, prompt_count, true, false) != prompt_count) {
        llama_error(options, "llama: prompt tokenization changed or failed\n");
        goto cleanup;
    }

    struct llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = (uint32_t)prompt_count + (uint32_t)options->tokens;
    if (context_params.n_ctx < 32) {
        context_params.n_ctx = 32;
    }
    context_params.n_batch = prompt_count < 512 ? (uint32_t)prompt_count : 512;
    context_params.n_ubatch = context_params.n_batch;
    context_params.n_seq_max = 1;
    context_params.n_threads = options->threads;
    context_params.n_threads_batch = options->threads;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    context_params.offload_kqv = options->webgpu;
    context_params.op_offload = options->webgpu;
    context_params.no_perf = true;
    context_params.abort_callback = options->service ? llama_service_abort : NULL;
    context_params.abort_callback_data = (void *)options;
    context = llama_init_from_model(model, context_params);
    if (recovery) recovery->context = context;
    if (context == NULL) {
        llama_error(options, "llama: could not create decoder context\n");
        goto cleanup;
    }
    if (llama_n_ctx(context) < (uint32_t)prompt_count + (uint32_t)options->tokens) {
        llama_error(options, "llama: allocated context is too small\n");
        goto cleanup;
    }

#ifdef COSMO_WEBGPU_BACKEND
    if (options->webgpu) {
        graphs_before = cosmo_webgpu_graph_count();
        dispatches_before = cosmo_webgpu_dispatch_count();
        matmuls_before = cosmo_webgpu_matmul_dispatch_count();
        submissions_before = cosmo_webgpu_submission_count();
        readbacks_before = cosmo_webgpu_readback_count();
    }
#endif
    for (int32_t offset = 0; offset < prompt_count;) {
        if (llama_cancelled(options)) { result = COSMO_LLAMA_CANCELLED; goto cleanup; }
        int32_t count = prompt_count - offset;
        if ((uint32_t)count > context_params.n_batch) {
            count = (int32_t)context_params.n_batch;
        }
        const struct llama_batch batch = llama_batch_get_one(prompt_tokens + offset, count);
        const int32_t status = llama_decode(context, batch);
        if (llama_cancelled(options)) { result = COSMO_LLAMA_CANCELLED; goto cleanup; }
        if (status != 0) {
            llama_error(options, "llama: prompt prefill failed: %" PRId32 "\n", status);
            goto cleanup;
        }
        offset += count;
    }

#ifdef COSMO_WEBGPU_BACKEND
    if (options->webgpu) {
        llama_synchronize(context);
        decode_graphs_before = cosmo_webgpu_graph_count();
        decode_dispatches_before = cosmo_webgpu_dispatch_count();
        decode_matmuls_before = cosmo_webgpu_matmul_dispatch_count();
        decode_submissions_before = cosmo_webgpu_submission_count();
        decode_readbacks_before = cosmo_webgpu_readback_count();
        if (decode_graphs_before == graphs_before || decode_dispatches_before == dispatches_before ||
            decode_matmuls_before == matmuls_before || decode_submissions_before == submissions_before) {
            llama_error(options, "llama: prompt prefill did not execute WebGPU matrix operations\n");
            goto cleanup;
        }
    }
#endif

    if (!options->selftest && !options->output &&
        fwrite(options->prompt, 1, prompt_bytes, stdout) != prompt_bytes) {
        llama_error(options, "llama: could not write prompt\n");
        goto cleanup;
    }
    const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
    for (int step = 0; step < options->tokens; ++step) {
        if (llama_cancelled(options)) { result = COSMO_LLAMA_CANCELLED; goto cleanup; }
        llama_token token;
        if (llama_greedy_checked(context, vocabulary_size, &token, options) != 0) {
            goto cleanup;
        }
        if (!options->selftest && llama_vocab_is_eog(vocab, token)) {
            break;
        }
        if (options->selftest || options->report_tokens) {
            if (step >= LLAMA_MAX_GENERATED) {
                llama_error(options, "llama: invalid self-test token count\n");
                goto cleanup;
            }
            selftest_ids[step] = token;
        }
        if (!options->selftest && llama_write_piece(vocab, token, options) != 0) {
            goto cleanup;
        }
        if (options->output) options->output->token_ids[generated] = token;
        ++generated;
        if (options->output) options->output->token_count = generated;

        /* Decode every selected token, including the final self-test token. */
        const struct llama_batch batch = llama_batch_get_one(&token, 1);
        const int32_t status = llama_decode(context, batch);
        if (llama_cancelled(options)) { result = COSMO_LLAMA_CANCELLED; goto cleanup; }
        if (status != 0) {
            llama_error(options, "llama: decode step %d failed: %" PRId32 "\n",
                    step + 1, status);
            goto cleanup;
        }
        ++decode_steps;
        if (options->service && options->service->progress)
            options->service->progress(options->service->user, generated, options->tokens);
    }
    llama_token next_token;
    if (llama_greedy_checked(context, vocabulary_size, &next_token, options) != 0) {
        goto cleanup;
    }
    if (options->selftest) {
        if (generated != LLAMA_SELFTEST_STEPS || decode_steps != LLAMA_SELFTEST_STEPS) {
            llama_error(options, "llama: self-test did not complete all decode steps\n");
            goto cleanup;
        }
        fputs(options->webgpu ? "WEBGPU_LLAMA_GREEDY_IDS " : "LLAMA_GREEDY_IDS ", stdout);
        for (int i = 0; i < generated; ++i) {
            printf("%s%" PRId32, i ? "," : "", selftest_ids[i]);
        }
        putchar('\n');
        for (int i = 0; i < generated; ++i) {
            if (selftest_ids[i] != llama_expected_ids[i]) {
                fprintf(stderr, "LLAMA_SELFTEST token %d mismatch: expected %" PRId32
                        ", actual %" PRId32 "\n", i,
                        llama_expected_ids[i], selftest_ids[i]);
                goto cleanup;
            }
        }
        printf("%s prompt_tokens=%" PRId32
               " decode_steps=%d vocab=%" PRId32 " finite_logits=1 threads=%d\n",
               options->webgpu ? "WEBGPU_LLAMA_SELFTEST" : "LLAMA_SELFTEST",
               prompt_count, decode_steps, vocabulary_size, options->threads);
    } else if (!options->output) {
        putchar('\n');
        llama_error(options, "llama: generated %d tokens using the linked %s backend\n",
                generated, options->webgpu ? "WebGPU" : "CPU");
        if (options->report_tokens) {
            llama_error(options, "LLAMA_GENERATED_IDS ");
            for (int i = 0; i < generated; ++i)
                fprintf(stderr, "%s%" PRId32, i ? "," : "", selftest_ids[i]);
            fputc('\n', stderr);
            fprintf(stderr, "LLAMA_GENERATION prompt_tokens=%" PRId32
                    " decode_steps=%d vocab=%" PRId32 " finite_logits=1 threads=%d\n",
                    prompt_count, decode_steps, vocabulary_size, options->threads);
        }
    }
#ifdef COSMO_WEBGPU_BACKEND
    if (options->webgpu) {
        const uint64_t graphs = cosmo_webgpu_graph_count() - graphs_before;
        const uint64_t dispatches = cosmo_webgpu_dispatch_count() - dispatches_before;
        const uint64_t matmuls = cosmo_webgpu_matmul_dispatch_count() - matmuls_before;
        const uint64_t submissions = cosmo_webgpu_submission_count() - submissions_before;
        const uint64_t readbacks = cosmo_webgpu_readback_count() - readbacks_before;
        const uint64_t decode_graphs = cosmo_webgpu_graph_count() - decode_graphs_before;
        const uint64_t decode_dispatches = cosmo_webgpu_dispatch_count() - decode_dispatches_before;
        const uint64_t decode_matmuls = cosmo_webgpu_matmul_dispatch_count() - decode_matmuls_before;
        const uint64_t decode_submissions = cosmo_webgpu_submission_count() - decode_submissions_before;
        const uint64_t decode_readbacks = cosmo_webgpu_readback_count() - decode_readbacks_before;
        const unsigned long native_opens = cosmo_webgpu_native_loader_open_count();
        const char *provider = cosmo_webgpu_provider_name();
        const int software = cosmo_webgpu_adapter_is_software();
        if (!graphs || !dispatches || !matmuls || !submissions || !readbacks ||
            selected_info == NULL || provider == NULL || strcmp(provider, selected_info->provider) ||
            software != selected_info->software) {
            llama_error(options, "llama: requested WebGPU device execution was not established\n");
            goto cleanup;
        }
        if (options->selftest && (native_opens || software != 1 || strcmp(provider, "embedded"))) {
            llama_error(options, "llama: self-test requires the embedded software WebGPU provider\n");
            goto cleanup;
        }
        if (decode_steps && (!decode_graphs || !decode_dispatches || !decode_matmuls ||
                             !decode_submissions || !decode_readbacks)) {
            llama_error(options, "llama: token decoding did not execute WebGPU matrix operations and readback\n");
            goto cleanup;
        }
        fprintf(options->selftest ? stdout : stderr,
                "WEBGPU_LLAMA_EXECUTION provider=%s software=%d graphs=%" PRIu64
                " submissions=%" PRIu64 " dispatches=%" PRIu64 " matmuls=%" PRIu64
                " readbacks=%" PRIu64 " native_loader_opens=%lu\n",
                provider, software, graphs, submissions, dispatches, matmuls, readbacks, native_opens);
        fprintf(options->selftest ? stdout : stderr,
                "WEBGPU_LLAMA_DECODE_EXECUTION steps=%d graphs=%" PRIu64
                " submissions=%" PRIu64 " dispatches=%" PRIu64 " matmuls=%" PRIu64
                " readbacks=%" PRIu64 "\n",
                decode_steps, decode_graphs, decode_submissions, decode_dispatches,
                decode_matmuls, decode_readbacks);
    }
#endif
    if (!options->output && (fflush(stdout) != 0 || ferror(stdout))) {
        llama_error(options, "llama: output write failed\n");
        goto cleanup;
    }
    result = 0;

cleanup:
    if (result == COSMO_LLAMA_CANCELLED && options->output) {
        options->output->cancelled = 1;
        snprintf(options->output->error, sizeof(options->output->error), "Text inference cancelled");
    }
    if (options->output) {
        cosmo_llama_service_cleanup(options->output);
    } else {
        if (context != NULL) {
            llama_free(context);
        }
        if (model != NULL) {
            llama_model_free(model);
        }
        free(prompt_tokens);
    }
    if (backend_initialized && !options->service) {
        llama_backend_free();
    }
    if (result == 0 && options->selftest &&
        (puts(options->webgpu ? "WEBGPU_LLAMA_SELFTEST PASS" : "LLAMA_SELFTEST PASS") == EOF || fflush(stdout) != 0)) {
        llama_error(options, "llama: self-test result write failed\n");
        return 1;
    }
    return result;
}

int cosmo_llama_selftest(void) {
    const char *model_path = LLAMA_DEMO_MODEL;
#ifdef COSMO_LLAMA_REFERENCE_BUILD
    const char *reference_fixture = getenv("COSMO_LLAMA_FIXTURE");
    if (reference_fixture != NULL && reference_fixture[0] != '\0') {
        model_path = reference_fixture;
    }
#endif
    const struct llama_run_options options = {
        model_path, LLAMA_DEMO_PROMPT, LLAMA_SELFTEST_STEPS, 1, true, false, "auto", false, NULL, NULL
    };
    return llama_run(&options);
}

int cosmo_llama_webgpu_selftest(void) {
    const struct llama_run_options options = {
        LLAMA_DEMO_MODEL, LLAMA_DEMO_PROMPT, LLAMA_SELFTEST_STEPS, 1, true, true, "auto", false, NULL, NULL
    };
    return llama_run(&options);
}

int cosmo_llama_generate(int argc, char **argv) {
    struct llama_run_options options = {
        LLAMA_DEMO_MODEL, LLAMA_DEMO_PROMPT, 64, 1, false, false, "auto", false, NULL, NULL
    };
#ifndef COSMO_LLAMA_REFERENCE_BUILD
    options.device = cosmo_config_device();
    options.webgpu = !strcmp(cosmo_config_backend(), "webgpu");
#endif
    for (int i = 1; i < argc; ++i) {
        const char *argument = argv[i];
        if (!strcmp(argument, "--report-tokens")) {
            options.report_tokens = true;
            continue;
        }
        if (!strcmp(argument, "--help") || !strcmp(argument, "-h")) {
            llama_usage();
            return 0;
        }
        if (strcmp(argument, "--model") && strcmp(argument, "--prompt") &&
            strcmp(argument, "--tokens") && strcmp(argument, "--threads") &&
            strcmp(argument, "--backend") && strcmp(argument, "--device")) {
            fprintf(stderr, "llama: unknown argument: %s\n", argument);
            return 2;
        }
        if (++i >= argc) {
            fprintf(stderr, "llama: missing value for %s\n", argument);
            return 2;
        }
        if (!strcmp(argument, "--model")) {
            if (argv[i][0] == '\0') {
                fputs("llama: model path must not be empty\n", stderr);
                return 2;
            }
            options.model_path = argv[i];
        } else if (!strcmp(argument, "--prompt")) {
            options.prompt = argv[i];
        } else if (!strcmp(argument, "--tokens")) {
            if (llama_parse_positive(argv[i], LLAMA_MAX_GENERATED, &options.tokens)) {
                fputs("llama: --tokens must be an integer from 1 to 4096\n", stderr);
                return 2;
            }
        } else if (!strcmp(argument, "--device")) {
            if (!*argv[i]) {
                fputs("llama: --device must not be empty\n", stderr);
                return 2;
            }
            options.device = argv[i];
        } else if (!strcmp(argument, "--backend")) {
            if (strcmp(argv[i], "cpu") && strcmp(argv[i], "webgpu")) {
                fputs("llama: --backend must be cpu or webgpu\n", stderr);
                return 2;
            }
            options.webgpu = !strcmp(argv[i], "webgpu");
        } else if (llama_parse_positive(argv[i], LLAMA_MAX_THREADS, &options.threads)) {
            fputs("llama: --threads must be an integer from 1 to 256\n", stderr);
            return 2;
        }
    }
    return llama_run(&options);
}

int cosmo_llama_service_engine(const struct cosmo_llama_request *request,
                              struct cosmo_llama_result *output) {
    if (request == NULL || output == NULL) return COSMO_LLAMA_INVALID;
    const char *backend = request->backend ? request->backend : "cpu";
    const char *model = request->model_path ? request->model_path : LLAMA_DEMO_MODEL;
    if (request->max_tokens < 1 || request->max_tokens > LLAMA_MAX_GENERATED ||
        request->threads < 1 || request->threads > LLAMA_MAX_THREADS ||
        (strcmp(backend, "cpu") && strcmp(backend, "webgpu")) ||
        !*model || strnlen(model, 32769) > 32768 ||
        (request->device && !*request->device)) {
        snprintf(output->error, sizeof(output->error), "Invalid text model, backend, device, token or thread limits");
        return COSMO_LLAMA_INVALID;
    }
    output->text = calloc(1, 1);
    output->token_ids = calloc((size_t)request->max_tokens, sizeof(*output->token_ids));
    if (output->text == NULL || output->token_ids == NULL) {
        snprintf(output->error, sizeof(output->error), "Could not allocate text inference result");
        return COSMO_LLAMA_ERROR;
    }
    const struct llama_run_options options = {
        model, request->prompt ? request->prompt : LLAMA_DEMO_PROMPT,
        request->max_tokens, request->threads, false, !strcmp(backend, "webgpu"),
        request->device ? request->device : "auto", false, request, output
    };
    return llama_run(&options);
}
