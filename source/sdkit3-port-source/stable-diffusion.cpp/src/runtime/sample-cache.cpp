#include "runtime/sample-cache.h"

namespace sd_sample {

    static float get_cache_reuse_threshold(const sd_cache_params_t& params) {
        float reuse_threshold = params.reuse_threshold;
        if (reuse_threshold == INFINITY) {
            if (params.mode == SD_CACHE_EASYCACHE) {
                reuse_threshold = 0.2f;
            } else if (params.mode == SD_CACHE_TEACACHE) {
                reuse_threshold = 0.2f;
            } else if (params.mode == SD_CACHE_UCACHE) {
                reuse_threshold = 1.0f;
            }
        }
        return std::max(0.0f, reuse_threshold);
    }

    bool SampleCacheRuntime::easycache_enabled() const {
        return mode == SampleCacheMode::EASYCACHE;
    }

    bool SampleCacheRuntime::teacache_enabled() const {
        return mode == SampleCacheMode::TEACACHE;
    }

    bool SampleCacheRuntime::ucache_enabled() const {
        return mode == SampleCacheMode::UCACHE;
    }

    bool SampleCacheRuntime::cachedit_enabled() const {
        return mode == SampleCacheMode::CACHEDIT;
    }

    static bool has_valid_cache_percent_range(const sd_cache_params_t& cache_params) {
        if (cache_params.mode != SD_CACHE_EASYCACHE &&
            cache_params.mode != SD_CACHE_TEACACHE &&
            cache_params.mode != SD_CACHE_UCACHE) {
            return true;
        }

        return cache_params.start_percent >= 0.0f &&
               cache_params.start_percent < 1.0f &&
               cache_params.end_percent > 0.0f &&
               cache_params.end_percent <= 1.0f &&
               cache_params.start_percent < cache_params.end_percent;
    }

    static void init_easycache_runtime(SampleCacheRuntime& runtime,
                                       SDVersion version,
                                       const sd_cache_params_t& cache_params,
                                       Denoiser* denoiser) {
        if (!sd_version_is_dit(version)) {
            LOG_WARN("EasyCache requested but not supported for this model type");
            return;
        }

        EasyCacheConfig config;
        config.enabled         = true;
        config.reuse_threshold = get_cache_reuse_threshold(cache_params);
        config.start_percent   = cache_params.start_percent;
        config.end_percent     = cache_params.end_percent;

        runtime.easycache.init(config, denoiser);
        if (!runtime.easycache.enabled()) {
            LOG_WARN("EasyCache requested but could not be initialized for this run");
            return;
        }

        runtime.mode = SampleCacheMode::EASYCACHE;
        LOG_INFO("EasyCache enabled - threshold: %.3f, start: %.2f, end: %.2f",
                 config.reuse_threshold,
                 config.start_percent,
                 config.end_percent);
    }

    static void init_teacache_runtime(SampleCacheRuntime& runtime,
                                      SDVersion version,
                                      const std::string& model_desc,
                                      const sd_cache_params_t& cache_params,
                                      const std::vector<float>& sigmas) {
        if (!sd_version_is_dit(version)) {
            LOG_WARN("TeaCache requested but not supported for this model type (only DiT models)");
            return;
        }

        TeaCacheConfig config;
        config.enabled       = true;
        config.start_percent = cache_params.start_percent;
        config.end_percent   = cache_params.end_percent;
        config.total_steps   = sigmas.size() > 1 ? static_cast<int>(sigmas.size() - 1) : 0;
        config.model_variant = model_desc;

        if (sd_version_is_ltxav(version)) {
            config.reuse_threshold = cache_params.reuse_threshold == INFINITY ? 0.05f
                                                                              : get_cache_reuse_threshold(cache_params);
            config.coefficients = {2.14700694e+01, -1.28016453e+01, 2.31279151e+00,
                                   7.92487521e-01, 9.69274326e-03};
        } else if (sd_version_is_wan(version)) {
            config.reuse_threshold = cache_params.reuse_threshold == INFINITY ? 0.20f
                                                                              : get_cache_reuse_threshold(cache_params);
            if (model_desc.find("1.3B") != std::string::npos) {
                config.coefficients = {2.39676752e+03, -1.31110545e+03, 2.01331979e+02,
                                       -8.29855975e+00, 1.37887774e-01};
            } else if (model_desc.find("I2V") != std::string::npos) {
                // Official Wan 2.1 I2V 480p coefficients. The threshold is
                // adjustable for other resolutions and compatible variants.
                config.coefficients = {-3.02331670e+02, 2.23948934e+02, -5.25463970e+01,
                                       5.87348440e+00, -2.01973289e-01};
            } else {
                config.coefficients = {-5784.54975374, 5449.50911966, -1811.16591783,
                                       256.27178429, -13.02252404};
            }
        } else {
            config.reuse_threshold = get_cache_reuse_threshold(cache_params);
        }

        runtime.teacache.init(config);
        if (!runtime.teacache.enabled()) {
            LOG_WARN("TeaCache requested but could not be initialized for this run");
            return;
        }
        runtime.mode = SampleCacheMode::TEACACHE;
        LOG_INFO("TeaCache enabled - model: %s, threshold: %.3f, start: %.2f, end: %.2f",
                 config.model_variant.c_str(), config.reuse_threshold,
                 config.start_percent, config.end_percent);
    }

    static void init_ucache_runtime(SampleCacheRuntime& runtime,
                                    SDVersion version,
                                    const sd_cache_params_t& cache_params,
                                    Denoiser* denoiser,
                                    const std::vector<float>& sigmas) {
        if (!sd_version_is_unet(version)) {
            LOG_WARN("UCache requested but not supported for this model type (only UNET models)");
            return;
        }

        UCacheConfig config;
        config.enabled                = true;
        config.reuse_threshold        = get_cache_reuse_threshold(cache_params);
        config.start_percent          = cache_params.start_percent;
        config.end_percent            = cache_params.end_percent;
        config.error_decay_rate       = std::max(0.0f, std::min(1.0f, cache_params.error_decay_rate));
        config.use_relative_threshold = cache_params.use_relative_threshold;
        config.reset_error_on_compute = cache_params.reset_error_on_compute;

        runtime.ucache.init(config, denoiser);
        if (!runtime.ucache.enabled()) {
            LOG_WARN("UCache requested but could not be initialized for this run");
            return;
        }

        runtime.ucache.set_sigmas(sigmas);
        runtime.mode = SampleCacheMode::UCACHE;
        LOG_INFO("UCache enabled - threshold: %.3f, start: %.2f, end: %.2f, decay: %.2f, relative: %s, reset: %s",
                 config.reuse_threshold,
                 config.start_percent,
                 config.end_percent,
                 config.error_decay_rate,
                 config.use_relative_threshold ? "true" : "false",
                 config.reset_error_on_compute ? "true" : "false");
    }

    static void init_cachedit_runtime(SampleCacheRuntime& runtime,
                                      SDVersion version,
                                      const sd_cache_params_t& cache_params,
                                      const std::vector<float>& sigmas) {
        if (!sd_version_is_dit(version)) {
            LOG_WARN("CacheDIT requested but not supported for this model type (only DiT models)");
            return;
        }

        DBCacheConfig dbcfg;
        dbcfg.enabled                     = (cache_params.mode == SD_CACHE_DBCACHE || cache_params.mode == SD_CACHE_CACHE_DIT);
        dbcfg.Fn_compute_blocks           = cache_params.Fn_compute_blocks;
        dbcfg.Bn_compute_blocks           = cache_params.Bn_compute_blocks;
        dbcfg.residual_diff_threshold     = cache_params.residual_diff_threshold;
        dbcfg.max_warmup_steps            = cache_params.max_warmup_steps;
        dbcfg.max_cached_steps            = cache_params.max_cached_steps;
        dbcfg.max_continuous_cached_steps = cache_params.max_continuous_cached_steps;
        if (cache_params.scm_mask != nullptr && strlen(cache_params.scm_mask) > 0) {
            dbcfg.steps_computation_mask = parse_scm_mask(cache_params.scm_mask);
        }
        dbcfg.scm_policy_dynamic = cache_params.scm_policy_dynamic;

        TaylorSeerConfig tcfg;
        tcfg.enabled             = (cache_params.mode == SD_CACHE_TAYLORSEER || cache_params.mode == SD_CACHE_CACHE_DIT);
        tcfg.n_derivatives       = cache_params.taylorseer_n_derivatives;
        tcfg.skip_interval_steps = cache_params.taylorseer_skip_interval;

        runtime.cachedit.init(dbcfg, tcfg);
        if (!runtime.cachedit.enabled()) {
            LOG_WARN("CacheDIT requested but could not be initialized for this run");
            return;
        }

        runtime.cachedit.set_sigmas(sigmas);
        runtime.mode = SampleCacheMode::CACHEDIT;
        LOG_INFO("CacheDIT enabled - mode: %s, Fn: %d, Bn: %d, threshold: %.3f, warmup: %d",
                 cache_params.mode == SD_CACHE_CACHE_DIT ? "DBCache+TaylorSeer" : (cache_params.mode == SD_CACHE_DBCACHE ? "DBCache" : "TaylorSeer"),
                 dbcfg.Fn_compute_blocks,
                 dbcfg.Bn_compute_blocks,
                 dbcfg.residual_diff_threshold,
                 dbcfg.max_warmup_steps);
    }

    static void init_spectrum_runtime(SampleCacheRuntime& runtime,
                                      SDVersion version,
                                      const sd_cache_params_t& cache_params,
                                      const std::vector<float>& sigmas) {
        if (!sd_version_is_unet(version) && !sd_version_is_dit(version)) {
            LOG_WARN("Spectrum requested but not supported for this model type (only UNET and DiT models)");
            return;
        }

        SpectrumConfig config;
        config.w            = cache_params.spectrum_w;
        config.m            = cache_params.spectrum_m;
        config.lam          = cache_params.spectrum_lam;
        config.window_size  = cache_params.spectrum_window_size;
        config.flex_window  = cache_params.spectrum_flex_window;
        config.warmup_steps = cache_params.spectrum_warmup_steps;
        config.stop_percent = cache_params.spectrum_stop_percent;

        size_t total_steps = sigmas.size() > 0 ? sigmas.size() - 1 : 0;
        runtime.spectrum.init(config, total_steps);
        runtime.spectrum_enabled = true;

        LOG_INFO("Spectrum enabled - w: %.2f, m: %d, lam: %.2f, window: %d, flex: %.2f, warmup: %d, stop: %.0f%%",
                 config.w, config.m, config.lam,
                 config.window_size, config.flex_window,
                 config.warmup_steps, config.stop_percent * 100.0f);
    }

    SampleCacheRuntime init_sample_cache_runtime(SDVersion version,
                                                 const sd_cache_params_t* cache_params,
                                                 Denoiser* denoiser,
                                                 const std::vector<float>& sigmas,
                                                 const std::string& model_desc) {
        SampleCacheRuntime runtime;
        if (cache_params == nullptr || cache_params->mode == SD_CACHE_DISABLED) {
            return runtime;
        }

        if (!has_valid_cache_percent_range(*cache_params)) {
            LOG_WARN("Cache disabled due to invalid percent range (start=%.3f, end=%.3f)",
                     cache_params->start_percent,
                     cache_params->end_percent);
            return runtime;
        }

        switch (cache_params->mode) {
            case SD_CACHE_EASYCACHE:
                init_easycache_runtime(runtime, version, *cache_params, denoiser);
                break;
            case SD_CACHE_TEACACHE:
                init_teacache_runtime(runtime, version, model_desc, *cache_params, sigmas);
                break;
            case SD_CACHE_UCACHE:
                init_ucache_runtime(runtime, version, *cache_params, denoiser, sigmas);
                break;
            case SD_CACHE_DBCACHE:
            case SD_CACHE_TAYLORSEER:
            case SD_CACHE_CACHE_DIT:
                init_cachedit_runtime(runtime, version, *cache_params, sigmas);
                break;
            case SD_CACHE_SPECTRUM:
                init_spectrum_runtime(runtime, version, *cache_params, sigmas);
                break;
            default:
                break;
        }

        return runtime;
    }

    SampleStepCacheDispatcher::SampleStepCacheDispatcher(SampleCacheRuntime& runtime, int step, float sigma)
        : runtime(runtime), step(step), sigma(sigma), step_index(step > 0 ? (step - 1) : -1) {
        if (step_index < 0) {
            return;
        }

        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                runtime.easycache.begin_step(step_index, sigma);
                break;
            case SampleCacheMode::TEACACHE:
                runtime.teacache.begin_step(step_index, sigma);
                break;
            case SampleCacheMode::UCACHE:
                runtime.ucache.begin_step(step_index, sigma);
                break;
            case SampleCacheMode::CACHEDIT:
                runtime.cachedit.begin_step(step_index, sigma);
                break;
            case SampleCacheMode::NONE:
                break;
        }
    }

    bool SampleStepCacheDispatcher::before_condition(const void* condition,
                                                     const sd::Tensor<float>& input,
                                                     sd::Tensor<float>* output) {
        if (step_index < 0 || condition == nullptr || output == nullptr) {
            return false;
        }

        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                return runtime.easycache.before_condition(condition, input, output, sigma, step_index);
            case SampleCacheMode::TEACACHE:
                return runtime.teacache.before_condition(condition, input, output, sigma, step_index);
            case SampleCacheMode::UCACHE:
                return runtime.ucache.before_condition(condition, input, output, sigma, step_index);
            case SampleCacheMode::CACHEDIT:
                return runtime.cachedit.before_condition(condition, input, output, sigma, step_index);
            case SampleCacheMode::NONE:
                return false;
        }

        return false;
    }

    void SampleStepCacheDispatcher::after_condition(const void* condition,
                                                    const sd::Tensor<float>& input,
                                                    const sd::Tensor<float>& output) {
        if (step_index < 0 || condition == nullptr) {
            return;
        }

        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                runtime.easycache.after_condition(condition, input, output);
                break;
            case SampleCacheMode::TEACACHE:
                runtime.teacache.after_condition(condition, input, output);
                break;
            case SampleCacheMode::UCACHE:
                runtime.ucache.after_condition(condition, input, output);
                break;
            case SampleCacheMode::CACHEDIT:
                runtime.cachedit.after_condition(condition, input, output);
                break;
            case SampleCacheMode::NONE:
                break;
        }
    }

    bool SampleStepCacheDispatcher::is_step_skipped() const {
        switch (runtime.mode) {
            case SampleCacheMode::EASYCACHE:
                return runtime.easycache.is_step_skipped();
            case SampleCacheMode::TEACACHE:
                return runtime.teacache.is_step_skipped();
            case SampleCacheMode::UCACHE:
                return runtime.ucache.is_step_skipped();
            case SampleCacheMode::CACHEDIT:
                return runtime.cachedit.is_step_skipped();
            case SampleCacheMode::NONE:
                return false;
        }

        return false;
    }

    void log_sample_cache_summary(const SampleCacheRuntime& runtime, size_t total_steps) {
        if (runtime.teacache_enabled()) {
            LOG_INFO("TeaCache reused %d condition evaluations; computed %d",
                     runtime.teacache.cache_hits,
                     runtime.teacache.model_evaluations);
        }
        if (runtime.easycache_enabled()) {
            if (runtime.easycache.total_steps_skipped > 0 && total_steps > 0) {
                if (runtime.easycache.total_steps_skipped < static_cast<int>(total_steps)) {
                    double speedup = static_cast<double>(total_steps) /
                                     static_cast<double>(total_steps - runtime.easycache.total_steps_skipped);
                    LOG_INFO("EasyCache skipped %d/%zu steps (%.2fx estimated speedup)",
                             runtime.easycache.total_steps_skipped,
                             total_steps,
                             speedup);
                } else {
                    LOG_INFO("EasyCache skipped %d/%zu steps",
                             runtime.easycache.total_steps_skipped,
                             total_steps);
                }
            } else if (total_steps > 0) {
                LOG_INFO("EasyCache completed without skipping steps");
            }
        }

        if (runtime.ucache_enabled()) {
            if (runtime.ucache.total_steps_skipped > 0 && total_steps > 0) {
                if (runtime.ucache.total_steps_skipped < static_cast<int>(total_steps)) {
                    double speedup = static_cast<double>(total_steps) /
                                     static_cast<double>(total_steps - runtime.ucache.total_steps_skipped);
                    LOG_INFO("UCache skipped %d/%zu steps (%.2fx estimated speedup)",
                             runtime.ucache.total_steps_skipped,
                             total_steps,
                             speedup);
                } else {
                    LOG_INFO("UCache skipped %d/%zu steps",
                             runtime.ucache.total_steps_skipped,
                             total_steps);
                }
            } else if (total_steps > 0) {
                LOG_INFO("UCache completed without skipping steps");
            }
        }

        if (runtime.cachedit_enabled()) {
            if (runtime.cachedit.total_steps_skipped > 0 && total_steps > 0) {
                if (runtime.cachedit.total_steps_skipped < static_cast<int>(total_steps)) {
                    double speedup = static_cast<double>(total_steps) /
                                     static_cast<double>(total_steps - runtime.cachedit.total_steps_skipped);
                    LOG_INFO("CacheDIT skipped %d/%zu steps (%.2fx estimated speedup)",
                             runtime.cachedit.total_steps_skipped,
                             total_steps,
                             speedup);
                } else {
                    LOG_INFO("CacheDIT skipped %d/%zu steps",
                             runtime.cachedit.total_steps_skipped,
                             total_steps);
                }
            } else if (total_steps > 0) {
                LOG_INFO("CacheDIT completed without skipping steps");
            }
        }

        if (runtime.spectrum_enabled && runtime.spectrum.total_steps_skipped > 0 && total_steps > 0) {
            double speedup = static_cast<double>(total_steps) /
                             static_cast<double>(total_steps - runtime.spectrum.total_steps_skipped);
            LOG_INFO("Spectrum skipped %d/%zu steps (%.2fx estimated speedup)",
                     runtime.spectrum.total_steps_skipped,
                     total_steps,
                     speedup);
        }
    }

}  // namespace sd_sample
