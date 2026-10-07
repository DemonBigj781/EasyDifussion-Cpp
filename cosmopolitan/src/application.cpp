/* SPDX-License-Identifier: MIT */
#include "application.h"
#include "application_services.h"
#include "server.h"
#include "config.hpp"
#include "logging.h"
#include "cosmo-webgpu.h"
#include "native_main_executor.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <system_error>

struct cosmo_app {
    std::shared_ptr<ApplicationServices> services;
    std::mutex mutex;
    std::unique_ptr<Server> server;
    std::atomic<bool> running{false};
    std::atomic<bool> used{false};
};

namespace {
void error_text(char *buffer, size_t bytes, const char *text) {
    if (buffer && bytes) std::snprintf(buffer, bytes, "%s", text);
}
void require_thread(int status, const char *operation) {
    if (status) throw std::system_error(status, std::generic_category(), operation);
}
struct HttpThread {
    Server *server;
    std::exception_ptr failure;
};
void *http_main(void *opaque) noexcept {
    auto &work = *static_cast<HttpThread *>(opaque);
    try { work.server->runTransport(); }
    catch (...) { work.failure = std::current_exception(); }
    return nullptr;
}
void run_frontends(cosmo_app *app, bool serve, cosmo_app_client client, void *user, int &result) {
    if (!serve) {
        try { if (client) result = client(app, user); }
        catch (...) { app->services->shutdown(); throw; }
        app->services->shutdown();
        return;
    }
    HttpThread work{app->server.get(), {}};
    pthread_attr_t attributes;
    require_thread(pthread_attr_init(&attributes), "application HTTP pthread_attr_init");
    const int stack_error = pthread_attr_setstacksize(&attributes, 8 * 1024 * 1024);
    pthread_t thread;
    const int create_error = stack_error ? stack_error : pthread_create(&thread, &attributes, http_main, &work);
    const int attribute_error = pthread_attr_destroy(&attributes);
    require_thread(create_error, "application HTTP pthread_create/stack");
    std::exception_ptr failure;
    try {
        require_thread(attribute_error, "application HTTP pthread_attr_destroy");
        work.server->waitTransportStarted();
        if (client) {
            result = client(app, user);
            work.server->stop();
        }
    } catch (...) {
        failure = std::current_exception();
        work.server->stop();
    }
    // The main-thread lane remains live through this join and the transport's
    // shared inference drain. Server and model destruction happen afterward.
    const int join_error = pthread_join(thread, nullptr);
    if (join_error) {
        std::fprintf(stderr, "Fatal application HTTP pthread_join error: %d\n", join_error);
        std::terminate();
    }
    if (failure) std::rethrow_exception(failure);
    if (work.failure) std::rethrow_exception(work.failure);
}
}

extern "C" cosmo_app *cosmo_app_create(char *error, size_t error_size) {
    error_text(error, error_size, "");
    try {
        const auto config = crow::json::load(cosmo_config_document());
        const auto &effective = config["effective"];
        set_log_level(std::string(effective["server"]["log_level"].s()));
        auto app = std::make_unique<cosmo_app>();
        auto index = std::make_shared<SdkitModelIndex>();
        index->setCheckpointDir(std::string(effective["models"]["checkpoint_dir"].s()));
        index->scanAllDirectories();
        ServerParams parameters;
        parameters.port = static_cast<int>(effective["server"]["port"].i());
        parameters.model_manager = index;
        const char *device = cosmo_config_sdkit_device();
        if (!device) throw std::runtime_error("Configured inference device is unavailable");
        parameters.compute_backend = device;
        app->services = std::make_shared<ApplicationServices>(parameters);
        return app.release();
    } catch (const std::exception &failure) {
        error_text(error, error_size, failure.what());
    } catch (...) { error_text(error, error_size, "Unknown application initialization failure"); }
    return nullptr;
}

extern "C" void cosmo_app_destroy(cosmo_app *app) { delete app; }
extern "C" void cosmo_app_reply_free(char *reply) { std::free(reply); }

extern "C" int cosmo_app_request(cosmo_app *app, cosmo_app_operation operation,
                                  const char *request_json, char **reply_json) {
    if (!reply_json) return 400;
    *reply_json = nullptr;
    try {
        if (!app) throw std::invalid_argument("Application is null");
        auto response = app->services->request(operation, request_json ? request_json : "{}");
        std::string body = response.body;
        if (!crow::json::load(body)) {
            crow::json::wvalue result;
            result["message"] = body;
            body = result.dump();
        }
        *reply_json = static_cast<char *>(std::malloc(body.size() + 1));
        if (!*reply_json) return 500;
        std::memcpy(*reply_json, body.c_str(), body.size() + 1);
        return response.code;
    } catch (const std::exception &failure) {
        try {
            crow::json::wvalue result; result["message"] = failure.what();
            const auto text = result.dump();
            *reply_json = static_cast<char *>(std::malloc(text.size() + 1));
            if (*reply_json) std::memcpy(*reply_json, text.c_str(), text.size() + 1);
        } catch (...) {}
        return 500;
    } catch (...) { return 500; }
}

extern "C" int cosmo_app_run(cosmo_app *app, int serve_http, cosmo_app_client client,
                              void *user, char *error, size_t error_size) {
    error_text(error, error_size, "");
    if (!app || (!serve_http && !client)) {
        error_text(error, error_size, "Application and at least one frontend are required");
        return 2;
    }
    if (app->used.exchange(true)) {
        error_text(error, error_size, "Application run lifecycle has already started");
        return 2;
    }
    app->running.store(true);
    int result = 0;
    try {
        if (serve_http) {
            std::lock_guard<std::mutex> lock(app->mutex);
            app->server = std::make_unique<Server>(app->services);
        }
        cosmo_native_main_service([&] { run_frontends(app, serve_http != 0, client, user, result); });
    } catch (const std::exception &failure) {
        error_text(error, error_size, failure.what()); result = 1;
    } catch (...) {
        error_text(error, error_size, "Unknown application service failure"); result = 1;
    }
    {
        std::lock_guard<std::mutex> lock(app->mutex);
        app->server.reset();
    }
    app->running.store(false);
    return result;
}

extern "C" void cosmo_app_stop(cosmo_app *app) {
    if (!app) return;
    app->services->closeAdmission();
    app->services->interrupt();
    std::lock_guard<std::mutex> lock(app->mutex);
    if (app->server) app->server->stop();
}
