#include "runtime.h"
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
#include <stdlib.h>
#include <string.h>

#define LLAMA_DEMO_MODEL "/zip/models/stories260K.gguf"
#define LLAMA_DEMO_PROMPT "Once upon a time"
#define LLAMA_SELFTEST_STEPS 16
#define LLAMA_MAX_PROMPT_BYTES (1024u * 1024u)
#define LLAMA_MAX_CONTEXT 32768u
#define LLAMA_MAX_GENERATED 4096
#define LLAMA_MAX_THREADS 256

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
};

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

static int llama_write_piece(const struct llama_vocab *vocab,
                             llama_token token) {
    char local[256];
    char *piece = local;
    int32_t length = llama_token_to_piece(vocab, token, local,
                                          (int32_t)sizeof(local), 0, false);
    if (length < 0) {
        if (length == INT32_MIN || -length > (int32_t)LLAMA_MAX_PROMPT_BYTES) {
            fputs("llama: token text exceeds the output buffer limit\n", stderr);
            return -1;
        }
        const int32_t capacity = -length;
        piece = malloc((size_t)capacity);
        if (piece == NULL) {
            fputs("llama: could not allocate token text\n", stderr);
            return -1;
        }
        length = llama_token_to_piece(vocab, token, piece, capacity, 0, false);
        if (length < 0 || length > capacity) {
            free(piece);
            fputs("llama: failed to render token text\n", stderr);
            return -1;
        }
    }
    const bool ok = fwrite(piece, 1, (size_t)length, stdout) == (size_t)length;
    if (piece != local) {
        free(piece);
    }
    if (!ok || fflush(stdout) != 0) {
        fputs("llama: failed to write generated text\n", stderr);
        return -1;
    }
    return 0;
}

static int llama_greedy_checked(struct llama_context *context,
                                int32_t vocabulary_size,
                                llama_token *selected) {
    const float *logits = llama_get_logits_ith(context, -1);
    if (logits == NULL || vocabulary_size <= 0) {
        fputs("llama: decoder produced no logits\n", stderr);
        return -1;
    }
    llama_token best = 0;
    for (int32_t i = 0; i < vocabulary_size; ++i) {
        if (!isfinite(logits[i])) {
            fprintf(stderr, "llama: non-finite logit at token %" PRId32 "\n", i);
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
        fputs("llama: WebGPU provider initialization failed\n", stderr);
        return 1;
    }
#else
    if (options->webgpu) {
        fputs("llama: WebGPU was not linked into this build\n", stderr);
        return 1;
    }
#endif
    const size_t prompt_bytes = strnlen(options->prompt,
                                       (size_t)LLAMA_MAX_PROMPT_BYTES + 1);

    if (prompt_bytes > LLAMA_MAX_PROMPT_BYTES || prompt_bytes > INT32_MAX) {
        fputs("llama: prompt exceeds the 1 MiB input limit\n", stderr);
        return 2;
    }

    /* All backend registrations are static; no plugin directory is scanned. */
    if (ggml_backend_reg_by_name("CPU") == NULL) {
        ggml_backend_reg_t cpu = ggml_backend_cpu_reg();
        if (cpu == NULL) {
            fputs("llama: linked CPU backend is unavailable\n", stderr);
            goto cleanup;
        }
        ggml_backend_register(cpu);
    }
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) == NULL) {
        fputs("llama: no registered CPU device\n", stderr);
        goto cleanup;
    }
    if (options->webgpu) {
        ggml_backend_reg_t webgpu = ggml_backend_reg_by_name("WebGPU");
        if (webgpu == NULL || ggml_backend_reg_dev_count(webgpu) == 0) {
            fputs("llama: requested WebGPU backend is unavailable\n", stderr);
            goto cleanup;
        }
#ifdef COSMO_WEBGPU_BACKEND
        if (cosmo_webgpu_select_device(requested_device)) {
            fprintf(stderr, "llama: requested WebGPU device is unavailable: %s\n", requested_device);
            goto cleanup;
        }
        selected_info = cosmo_webgpu_device_metadata("auto");
        if (selected_info == NULL) goto cleanup;
        selected_devices[0] = ggml_backend_dev_by_name(selected_info->selector);
        if (selected_devices[0] == NULL || ggml_backend_dev_backend_reg(selected_devices[0]) != webgpu) {
            fputs("llama: selected adapter has no matching GGML device\n", stderr);
            goto cleanup;
        }
        fprintf(stderr, "LLAMA_DEVICE selector=%s provider=%s software=%d stable_id=%s\n",
                selected_info->selector, selected_info->provider, selected_info->software,
                selected_info->stable_id && *selected_info->stable_id ? selected_info->stable_id : "unavailable");
#endif
    } else if (strcmp(requested_device, "auto") && strcmp(requested_device, "CPU") &&
               strcmp(requested_device, "cpu")) {
        fprintf(stderr, "llama: device %s does not belong to the CPU backend\n", requested_device);
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
    model_params.progress_callback = NULL;
    model = llama_model_load_from_file(options->model_path, model_params);
    if (model == NULL) {
        fprintf(stderr, "llama: could not load model: %s\n", options->model_path);
        goto cleanup;
    }
    if (llama_model_has_encoder(model) || !llama_model_has_decoder(model)) {
        fputs("llama: unsupported model: this entry point requires a "
              "decoder-only text model\n", stderr);
        goto cleanup;
    }
    vocab = llama_model_get_vocab(model);
    if (vocab == NULL || llama_vocab_n_tokens(vocab) <= 0) {
        fputs("llama: model has no usable vocabulary\n", stderr);
        goto cleanup;
    }

    const int32_t needed = llama_tokenize(vocab, options->prompt,
                                          (int32_t)prompt_bytes,
                                          NULL, 0, true, false);
    if (needed >= 0 || needed == INT32_MIN ||
        (uint32_t)(-needed) > LLAMA_MAX_CONTEXT ||
        (size_t)(-needed) > SIZE_MAX / sizeof(*prompt_tokens)) {
        fputs("llama: prompt tokenization is empty, invalid, or too large\n", stderr);
        goto cleanup;
    }
    prompt_count = -needed;
    if ((uint32_t)prompt_count > LLAMA_MAX_CONTEXT - (uint32_t)options->tokens) {
        fputs("llama: prompt plus generation exceeds 32768 context tokens\n", stderr);
        goto cleanup;
    }
    prompt_tokens = malloc((size_t)prompt_count * sizeof(*prompt_tokens));
    if (prompt_tokens == NULL) {
        fputs("llama: could not allocate prompt tokens\n", stderr);
        goto cleanup;
    }
    if (llama_tokenize(vocab, options->prompt, (int32_t)prompt_bytes,
                       prompt_tokens, prompt_count, true, false) != prompt_count) {
        fputs("llama: prompt tokenization changed or failed\n", stderr);
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
    context = llama_init_from_model(model, context_params);
    if (context == NULL) {
        fputs("llama: could not create decoder context\n", stderr);
        goto cleanup;
    }
    if (llama_n_ctx(context) < (uint32_t)prompt_count + (uint32_t)options->tokens) {
        fputs("llama: allocated context is too small\n", stderr);
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
        int32_t count = prompt_count - offset;
        if ((uint32_t)count > context_params.n_batch) {
            count = (int32_t)context_params.n_batch;
        }
        const struct llama_batch batch = llama_batch_get_one(prompt_tokens + offset, count);
        const int32_t status = llama_decode(context, batch);
        if (status != 0) {
            fprintf(stderr, "llama: prompt prefill failed: %" PRId32 "\n", status);
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
            fputs("llama: prompt prefill did not execute WebGPU matrix operations\n", stderr);
            goto cleanup;
        }
    }
#endif

    if (!options->selftest &&
        fwrite(options->prompt, 1, prompt_bytes, stdout) != prompt_bytes) {
        fputs("llama: could not write prompt\n", stderr);
        goto cleanup;
    }
    const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
    for (int step = 0; step < options->tokens; ++step) {
        llama_token token;
        if (llama_greedy_checked(context, vocabulary_size, &token) != 0) {
            goto cleanup;
        }
        if (!options->selftest && llama_vocab_is_eog(vocab, token)) {
            break;
        }
        if (options->selftest || options->report_tokens) {
            if (step >= LLAMA_MAX_GENERATED) {
                fputs("llama: invalid self-test token count\n", stderr);
                goto cleanup;
            }
            selftest_ids[step] = token;
        }
        if (!options->selftest && llama_write_piece(vocab, token) != 0) {
            goto cleanup;
        }
        ++generated;

        /* Decode every selected token, including the final self-test token. */
        const struct llama_batch batch = llama_batch_get_one(&token, 1);
        const int32_t status = llama_decode(context, batch);
        if (status != 0) {
            fprintf(stderr, "llama: decode step %d failed: %" PRId32 "\n",
                    step + 1, status);
            goto cleanup;
        }
        ++decode_steps;
    }
    llama_token next_token;
    if (llama_greedy_checked(context, vocabulary_size, &next_token) != 0) {
        goto cleanup;
    }
    if (options->selftest) {
        if (generated != LLAMA_SELFTEST_STEPS || decode_steps != LLAMA_SELFTEST_STEPS) {
            fputs("llama: self-test did not complete all decode steps\n", stderr);
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
    } else {
        putchar('\n');
        fprintf(stderr, "llama: generated %d tokens using the linked %s backend\n",
                generated, options->webgpu ? "WebGPU" : "CPU");
        if (options->report_tokens) {
            fputs("LLAMA_GENERATED_IDS ", stderr);
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
            fputs("llama: requested WebGPU device execution was not established\n", stderr);
            goto cleanup;
        }
        if (options->selftest && (native_opens || software != 1 || strcmp(provider, "embedded"))) {
            fputs("llama: self-test requires the embedded software WebGPU provider\n", stderr);
            goto cleanup;
        }
        if (decode_steps && (!decode_graphs || !decode_dispatches || !decode_matmuls ||
                             !decode_submissions || !decode_readbacks)) {
            fputs("llama: token decoding did not execute WebGPU matrix operations and readback\n", stderr);
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
    if (fflush(stdout) != 0 || ferror(stdout)) {
        fputs("llama: output write failed\n", stderr);
        goto cleanup;
    }
    result = 0;

cleanup:
    if (context != NULL) {
        llama_free(context);
    }
    if (model != NULL) {
        llama_model_free(model);
    }
    free(prompt_tokens);
    if (backend_initialized) {
        llama_backend_free();
    }
    if (result == 0 && options->selftest &&
        (puts(options->webgpu ? "WEBGPU_LLAMA_SELFTEST PASS" : "LLAMA_SELFTEST PASS") == EOF || fflush(stdout) != 0)) {
        fputs("llama: self-test result write failed\n", stderr);
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
        model_path, LLAMA_DEMO_PROMPT, LLAMA_SELFTEST_STEPS, 1, true, false, "auto", false
    };
    return llama_run(&options);
}

int cosmo_llama_webgpu_selftest(void) {
    const struct llama_run_options options = {
        LLAMA_DEMO_MODEL, LLAMA_DEMO_PROMPT, LLAMA_SELFTEST_STEPS, 1, true, true, "auto", false
    };
    return llama_run(&options);
}

int cosmo_llama_generate(int argc, char **argv) {
    struct llama_run_options options = {
        LLAMA_DEMO_MODEL, LLAMA_DEMO_PROMPT, 64, 1, false, false, "auto", false
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
