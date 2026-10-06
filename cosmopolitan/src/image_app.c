#ifdef __COSMOPOLITAN__
#include <cosmo.h>
#endif

#include "runtime.h"
#include "image_bridge.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "stable-diffusion.h"
#ifdef COSMO_WEBGPU_BACKEND
#include "cosmo-webgpu.h"
#endif

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* These C symbols are provided by the existing stable-diffusion miniz object.
   Its amalgamated header includes implementation, so do not include it here. */
extern void *tdefl_write_image_to_png_file_in_memory(const void *pixels, int width,
                                                   int height, int channels,
                                                   size_t *size);
extern void mz_free(void *memory);

struct image_counters {
    uint64_t graphs, submissions, dispatches, matmuls, readbacks;
};

static struct image_counters image_counters_now(void) {
    struct image_counters value = {0};
#ifdef COSMO_WEBGPU_BACKEND
    value.graphs = cosmo_webgpu_graph_count();
    value.submissions = cosmo_webgpu_submission_count();
    value.dispatches = cosmo_webgpu_dispatch_count();
    value.matmuls = cosmo_webgpu_matmul_dispatch_count();
    value.readbacks = cosmo_webgpu_readback_count();
#endif
    return value;
}

static struct image_counters image_counters_delta(struct image_counters after,
                                                  struct image_counters before) {
    after.graphs -= before.graphs;
    after.submissions -= before.submissions;
    after.dispatches -= before.dispatches;
    after.matmuls -= before.matmuls;
    after.readbacks -= before.readbacks;
    return after;
}

struct image_sampling {
    int first_step, last_step, total_steps, completed_callbacks, intervals;
    bool monotonic;
    struct image_counters previous, executed;
};

static void image_sample_progress(int step, int steps, float seconds, void *data) {
    struct image_sampling *sampling = data;
    const struct image_counters current = image_counters_now();
    if (step < 1 || steps < 1 || step > steps || !isfinite(seconds) || seconds < 0)
        sampling->monotonic = false;
    if (!sampling->completed_callbacks) {
        sampling->first_step = step;
        sampling->total_steps = steps;
        if (step != 1) sampling->monotonic = false;
    } else {
        if (steps != sampling->total_steps || step != sampling->last_step + 1)
            sampling->monotonic = false;
        if (current.graphs < sampling->previous.graphs ||
            current.submissions < sampling->previous.submissions ||
            current.dispatches < sampling->previous.dispatches ||
            current.matmuls < sampling->previous.matmuls ||
            current.readbacks < sampling->previous.readbacks) {
            sampling->monotonic = false;
        } else {
            const struct image_counters delta = image_counters_delta(current, sampling->previous);
            sampling->executed.graphs += delta.graphs;
            sampling->executed.submissions += delta.submissions;
            sampling->executed.dispatches += delta.dispatches;
            sampling->executed.matmuls += delta.matmuls;
            sampling->executed.readbacks += delta.readbacks;
        }
        ++sampling->intervals;
    }
    sampling->previous = current;
    sampling->last_step = step;
    ++sampling->completed_callbacks;
    fprintf(stderr, "\nIMAGE_SAMPLE_PROGRESS step=%d total_steps=%d seconds=%.6f\n",
            step, steps, (double)seconds);
    fflush(stderr);
}

static void image_print_sampling(bool webgpu, const struct image_sampling *sampling) {
    printf("IMAGE_SAMPLING backend=%s first_step=%d last_step=%d total_steps=%d"
           " completed_callbacks=%d intervals=%d monotonic=%d graphs=%" PRIu64
           " submissions=%" PRIu64 " dispatches=%" PRIu64 " matmuls=%" PRIu64
           " readbacks=%" PRIu64 "\n", webgpu ? "webgpu" : "cpu",
           sampling->first_step, sampling->last_step, sampling->total_steps,
           sampling->completed_callbacks, sampling->intervals, sampling->monotonic,
           sampling->executed.graphs, sampling->executed.submissions,
           sampling->executed.dispatches, sampling->executed.matmuls,
           sampling->executed.readbacks);
    fflush(stdout);
}

static void image_print_stage(const char *stage, bool webgpu, bool succeeded,
                              int64_t elapsed_us, struct image_counters value) {
    printf("IMAGE_%s backend=%s success=%d seconds=%.6f graphs=%" PRIu64
           " submissions=%" PRIu64 " dispatches=%" PRIu64 " matmuls=%" PRIu64
           " readbacks=%" PRIu64 "\n", stage, webgpu ? "webgpu" : "cpu",
           succeeded, (double)elapsed_us / 1000000.0, value.graphs,
           value.submissions, value.dispatches, value.matmuls, value.readbacks);
    fflush(stdout);
}

static void image_usage(void) {
    puts("Usage: image --model PATH --prompt TEXT --output PATH [OPTIONS]\n"
         "  --backend cpu|webgpu   Compute backend (default: cpu)\n"
         "  --width N --height N   Multiples of 64, 64..4096 (default: 512)\n"
         "  --steps N             Sampling steps, 1..1000 (default: 20)\n"
         "  --seed N              Reproducible seed, 0..INT64_MAX (default: 42)\n"
         "  --threads N           CPU threads, 1..256 (default: native core count)\n"
         "  --cfg-scale F         Finite guidance scale, 0..100 (default: 7)\n"
         "  --negative-prompt TEXT  Negative conditioning (default: empty)\n"
         "  --sampler NAME        Native sampler (default: model default)\n"
         "  --scheduler NAME      Native scheduler (default: model default)\n"
         "  --vae PATH            Optional separate VAE weights\n"
         "Saves one PNG. Model and output paths must be explicit.\n"
         "WebGPU uses embedded software Vulkan and permits GGML CPU fallback\n"
         "for unsupported operators; execution counters measure WebGPU work.\n"
         "Flash attention and SageAttention are disabled by default.");
}

static int image_parse_integer(const char *text, int64_t minimum,
                                int64_t maximum, int64_t *out) {
    char *end;
    if (!text || text[0] < '0' || text[0] > '9') return -1;
    errno = 0;
    const intmax_t value = strtoimax(text, &end, 10);
    if (errno || *end || value < minimum || value > maximum) return -1;
    *out = (int64_t)value;
    return 0;
}

static int image_parse_cfg(const char *text, float *out) {
    char *end;
    if (!text || !*text || isspace((unsigned char)text[0])) return -1;
    errno = 0;
    const float value = strtof(text, &end);
    if (errno || end == text || *end || !isfinite(value) || value < 0 || value > 100) return -1;
    *out = value;
    return 0;
}

static bool image_input_file(const char *path, struct stat *info) {
    if (stat(path, info) || !S_ISREG(info->st_mode) || info->st_size <= 0) {
        fprintf(stderr, "image: input is not a nonempty regular file: %s\n", path);
        return false;
    }
    return true;
}

static bool image_same_file(const struct stat *a, const struct stat *b) {
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino;
}

static int image_save_png(const char *path, const sd_image_t *image, size_t *written) {
    int result = -1;
    size_t png_size = 0;
    void *png = tdefl_write_image_to_png_file_in_memory(image->data,
            (int)image->width, (int)image->height, (int)image->channel, &png_size);
    char *temporary = NULL;
    bool temporary_created = false;
    FILE *file = NULL;
    int fd = -1;
    if (!png || !png_size) {
        fputs("image: PNG encoding failed\n", stderr);
        goto cleanup;
    }
    const size_t length = strlen(path);
    if (length > SIZE_MAX - 12 || !(temporary = malloc(length + 12))) {
        fputs("image: PNG output path allocation failed\n", stderr);
        goto cleanup;
    }
    snprintf(temporary, length + 12, "%s.tmp.XXXXXX", path);
    fd = mkstemp(temporary);
    temporary_created = fd >= 0;
    if (fd < 0 || !(file = fdopen(fd, "wb"))) {
        fprintf(stderr, "image: cannot create PNG beside %s: %s\n", path, strerror(errno));
        goto cleanup;
    }
    fd = -1;
    bool ok = fwrite(png, 1, png_size, file) == png_size;
    if (fflush(file)) ok = false;
    if (fclose(file)) ok = false;
    file = NULL;
    if (!ok) {
        fprintf(stderr, "image: failed to write complete PNG for %s\n", path);
        goto cleanup;
    }
    if (rename(temporary, path)) {
        fprintf(stderr, "image: PNG write failed for %s: %s\n", path, strerror(errno));
        goto cleanup;
    }
    *written = png_size;
    result = 0;
cleanup:
    if (file) fclose(file);
    if (fd >= 0) close(fd);
    if (temporary) {
        if (result && temporary_created) unlink(temporary);
        free(temporary);
    }
    if (png) mz_free(png);
    return result;
}

int cosmo_image_generate(int argc, char **argv) {
#ifdef __COSMOPOLITAN__
    ShowCrashReports();
#endif
    static const char *const names[] = {
        "--model", "--prompt", "--output", "--backend", "--width", "--height",
        "--steps", "--seed", "--threads", "--cfg-scale", "--negative-prompt",
        "--sampler", "--scheduler", "--vae"
    };
    bool seen[sizeof(names) / sizeof(names[0])] = {false};
    bool webgpu = false;
    const char *output = NULL;
    sd_ctx_params_t context_params;
    sd_img_gen_params_t params;
    sd_ctx_params_init(&context_params);
    sd_img_gen_params_init(&params);
    params.seed = 42;
    params.negative_prompt = "";
    if (context_params.n_threads < 1) context_params.n_threads = 1;
    if (context_params.n_threads > 256) context_params.n_threads = 256;
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        image_usage();
        return 0;
    }
    for (int i = 1; i < argc; i += 2) {
        size_t option = 0;
        while (option < sizeof(names) / sizeof(names[0]) && strcmp(argv[i], names[option])) ++option;
        if (option == sizeof(names) / sizeof(names[0]) || seen[option] || i + 1 == argc) {
            fprintf(stderr, "image: unknown, duplicate, or incomplete option: %s\n", argv[i]);
            return 2;
        }
        seen[option] = true;
        const char *value = argv[i + 1];
        int64_t number = 0;
        switch (option) {
            case 0: context_params.model_path = value; break;
            case 1: params.prompt = value; break;
            case 2: output = value; break;
            case 3:
                if (strcmp(value, "cpu") && strcmp(value, "webgpu")) goto bad_value;
                webgpu = !strcmp(value, "webgpu");
                break;
            case 4:
            case 5:
                if (image_parse_integer(value, 64, 4096, &number) || number % 64) goto bad_value;
                if (option == 4) params.width = (int)number;
                else params.height = (int)number;
                break;
            case 6:
                if (image_parse_integer(value, 1, 1000, &number)) goto bad_value;
                params.sample_params.sample_steps = (int)number;
                break;
            case 7:
                if (image_parse_integer(value, 0, INT64_MAX, &number)) goto bad_value;
                params.seed = number;
                break;
            case 8:
                if (image_parse_integer(value, 1, 256, &number)) goto bad_value;
                context_params.n_threads = (int)number;
                break;
            case 9:
                if (image_parse_cfg(value, &params.sample_params.guidance.txt_cfg)) goto bad_value;
                break;
            case 10: params.negative_prompt = value; break;
            case 11:
                params.sample_params.sample_method = str_to_sample_method(value);
                if (params.sample_params.sample_method == SAMPLE_METHOD_COUNT) goto bad_value;
                break;
            case 12:
                params.sample_params.scheduler = str_to_scheduler(value);
                if (params.sample_params.scheduler == SCHEDULER_COUNT) goto bad_value;
                break;
            case 13: context_params.vae_path = value; break;
        }
        continue;
bad_value:
        fprintf(stderr, "image: invalid value for %s: %s\n", argv[i], value);
        return 2;
    }
    if (!context_params.model_path || !*context_params.model_path || !seen[1] || !output || !*output ||
        (context_params.vae_path && !*context_params.vae_path)) {
        fputs("image: --model PATH, --prompt TEXT, and --output PATH are required\n", stderr);
        return 2;
    }
    if (strnlen(params.prompt, 1048577) > 1048576 || strnlen(params.negative_prompt, 1048577) > 1048576) {
        fputs("image: prompts must not exceed 1 MiB each\n", stderr);
        return 2;
    }
    struct stat model_info, vae_info, output_info;
    if (!image_input_file(context_params.model_path, &model_info) ||
        (context_params.vae_path && !image_input_file(context_params.vae_path, &vae_info))) return 1;
    if (!stat(output, &output_info) &&
        (!S_ISREG(output_info.st_mode) || image_same_file(&model_info, &output_info) ||
         (context_params.vae_path && image_same_file(&vae_info, &output_info)))) {
        fputs("image: output must be a regular file distinct from model/VAE inputs\n", stderr);
        return 2;
    }
#ifdef COSMO_WEBGPU_BACKEND
    if (cosmo_webgpu_initialize()) {
        fputs("image: embedded WebGPU provider initialization failed\n", stderr);
        return 1;
    }
#else
    if (webgpu) {
        fputs("image: WebGPU was not linked into this build\n", stderr);
        return 1;
    }
#endif
    ggml_backend_reg_t registry = ggml_backend_reg_by_name(webgpu ? "WebGPU" : "CPU");
    if (!registry || !ggml_backend_reg_dev_count(registry)) {
        fprintf(stderr, "image: requested %s backend is unavailable\n", webgpu ? "WebGPU" : "CPU");
        return 1;
    }
    context_params.backend = ggml_backend_dev_name(ggml_backend_reg_dev_get(registry, 0));
    context_params.flash_attn = false;
    context_params.diffusion_flash_attn = false;
    context_params.diffusion_sage_attn = false;
    int result = 1;
    sd_image_t *images = NULL;
    int image_count = 0;
    const struct image_counters before_load = image_counters_now();
    const int64_t load_start = ggml_time_us();
    char native_error[512];
    sd_ctx_t *context = cosmo_image_create_context(&context_params, native_error, sizeof(native_error));
    image_print_stage("MODEL_LOAD", webgpu, context != NULL, ggml_time_us() - load_start,
                      image_counters_delta(image_counters_now(), before_load));
    if (!context) {
        fprintf(stderr, "image: model loading failed%s%s\n", *native_error ? ": " : "", native_error);
        goto cleanup;
    }
    if (!sd_ctx_supports_image_generation(context)) {
        fputs("image: this model does not support image generation\n", stderr);
        goto cleanup;
    }
    if (params.sample_params.sample_method == SAMPLE_METHOD_COUNT)
        params.sample_params.sample_method = sd_get_default_sample_method(context);
    if (params.sample_params.scheduler == SCHEDULER_COUNT)
        params.sample_params.scheduler = sd_get_default_scheduler(context, params.sample_params.sample_method);
    printf("IMAGE_PARAMETERS width=%d height=%d steps=%d seed=%" PRId64
           " threads=%d cfg_scale=%.9g sampler=%s scheduler=%s cpu_fallback_allowed=%d\n",
           params.width, params.height, params.sample_params.sample_steps, params.seed,
           context_params.n_threads, (double)params.sample_params.guidance.txt_cfg,
           sd_sample_method_name(params.sample_params.sample_method),
           sd_scheduler_name(params.sample_params.scheduler), webgpu);
    fflush(stdout);
    const struct image_counters before_generate = image_counters_now();
    const int64_t generate_start = ggml_time_us();
    struct image_sampling sampling = {.monotonic = true};
    /* This public callback is only reported after completed sampling steps.
       Starting at its first call excludes CLIP and the first denoising step;
       ending at its last call excludes final VAE decoding. No preview work is
       enabled in this CLI, so intervening WebGPU work belongs to sampling. */
    sd_set_preview_callback(NULL, PREVIEW_NONE, 0, false, false, NULL);
    sd_set_sample_progress_callback(image_sample_progress, &sampling);
    const bool generated = cosmo_image_generate_pixels(context, &params, &images, &image_count,
                                                       native_error, sizeof(native_error));
    sd_set_sample_progress_callback(NULL, NULL);
    const struct image_counters executed = image_counters_delta(image_counters_now(), before_generate);
    image_print_stage("GENERATION", webgpu, generated, ggml_time_us() - generate_start, executed);
    image_print_sampling(webgpu, &sampling);
    if (!generated || !images || image_count != 1 || !images[0].data ||
        images[0].width != (uint32_t)params.width || images[0].height != (uint32_t)params.height ||
        images[0].channel < 1 || images[0].channel > 4) {
        fprintf(stderr, "image: generation failed or returned an invalid image%s%s\n",
                *native_error ? ": " : "", native_error);
        goto cleanup;
    }
#ifdef COSMO_WEBGPU_BACKEND
    if (webgpu) {
        const char *provider = cosmo_webgpu_provider_name();
        const int software = cosmo_webgpu_adapter_is_software();
        const unsigned long native_opens = cosmo_webgpu_native_loader_open_count();
        printf("IMAGE_WEBGPU_EXECUTION provider=%s software=%d native_loader_opens=%lu"
               " cpu_fallback_allowed=1 cpu_fallback_measured=0\n",
               provider ? provider : "unknown", software, native_opens);
        const char *adapter = cosmo_webgpu_adapter_name();
        printf("IMAGE_WEBGPU_ADAPTER %s\n", adapter ? adapter : "unknown");
        if (!provider || strcmp(provider, "embedded") || software != 1 || native_opens ||
            !executed.graphs || !executed.submissions || !executed.dispatches ||
            !executed.matmuls || !executed.readbacks) {
            fputs("image: generation did not establish requested embedded WebGPU execution\n", stderr);
            goto cleanup;
        }
    }
#endif
    const size_t pixel_bytes = (size_t)images[0].width * images[0].height * images[0].channel;
    unsigned minimum = 255, maximum = 0;
    uint64_t sum = 0;
    for (size_t i = 0; i < pixel_bytes; ++i) {
        const unsigned value = images[0].data[i];
        if (value < minimum) minimum = value;
        if (value > maximum) maximum = value;
        sum += value;
    }
    size_t png_bytes = 0;
    if (image_save_png(output, images, &png_bytes)) goto cleanup;
    printf("IMAGE_OUTPUT width=%u height=%u channels=%u pixel_bytes=%zu png_bytes=%zu"
           " pixel_min=%u pixel_max=%u pixel_mean=%.9f\n",
           images[0].width, images[0].height, images[0].channel, pixel_bytes, png_bytes,
           minimum, maximum, (double)sum / pixel_bytes);
    puts("IMAGE_INFERENCE PASS");
    if (fflush(stdout) || ferror(stdout)) {
        fputs("image: result logging failed\n", stderr);
        goto cleanup;
    }
    result = 0;
cleanup:
    free_sd_images(images, image_count);
    if (context) free_sd_ctx(context);
    return result;
}
