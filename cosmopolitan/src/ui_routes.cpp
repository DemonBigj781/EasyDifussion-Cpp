#include "ui_routes.hpp"
#include "easy_diffusion/ui/pages.hpp"

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string native_page_scripts(std::string html, const std::string &script) {
    const std::string prefix = "<script defer src=\"";
    size_t position = 0;
    while ((position = html.find(prefix, position)) != std::string::npos) {
        const size_t name_start = position + prefix.size();
        const size_t name_end = html.find('"', name_start);
        const size_t end = html.find("</script>", name_end);
        if (name_end == std::string::npos || end == std::string::npos) break;
        const std::string name = html.substr(name_start, name_end - name_start);
        if (name == "/cpp-ui/scripts/kiosk.js" || name == script) {
            position = end + 9;
        } else {
            html.erase(position, end + 9 - position);
        }
    }
    return html;
}

crow::response page(const std::string &path) {
    try {
        std::string html = easy_diffusion::ui::pages::render(path).render();
        if (path == "/") html = native_page_scripts(std::move(html), "/cpp-ui/scripts/generate.js");
        if (path == "/settings" || path == "/settings/gpu")
            html = native_page_scripts(std::move(html), "/cpp-ui/scripts/backend-platform.js");
        crow::response result(std::move(html));
        result.set_header("Content-Type", "text/html; charset=utf-8");
        return result;
    } catch (const std::out_of_range &) {
        return crow::response(404);
    }
}

crow::response resource(const std::string &prefix, const std::string &path) {
    if (path.empty() || path[0] == '/' || path.find("..") != std::string::npos ||
        path.find('\\') != std::string::npos || path.find('\0') != std::string::npos) return crow::response(400);
    const std::string name = "/zip/" + prefix + path;
    FILE *file = std::fopen(name.c_str(), "rb");
    if (!file) return crow::response(404);
    std::string body;
    char block[16384];
    size_t size;
    while ((size = std::fread(block, 1, sizeof(block), file)) != 0) body.append(block, size);
    const bool failed = std::ferror(file);
    std::fclose(file);
    if (failed) return crow::response(500);
    crow::response result(std::move(body));
    const auto dot = path.find_last_of('.');
    const std::string extension = dot == std::string::npos ? "" : path.substr(dot);
    const char *mime = "application/octet-stream";
    if (extension == ".css") mime = "text/css; charset=utf-8";
    else if (extension == ".js") mime = "text/javascript; charset=utf-8";
    else if (extension == ".json") mime = "application/json";
    else if (extension == ".html") mime = "text/html; charset=utf-8";
    else if (extension == ".svg") mime = "image/svg+xml";
    else if (extension == ".png") mime = "image/png";
    else if (extension == ".jpg" || extension == ".jpeg") mime = "image/jpeg";
    else if (extension == ".gif") mime = "image/gif";
    else if (extension == ".woff2") mime = "font/woff2";
    else if (extension == ".woff") mime = "font/woff";
    else if (extension == ".ttf") mime = "font/ttf";
    result.set_header("Content-Type", mime);
    result.set_header("X-Content-Type-Options", "nosniff");
    return result;
}

} // namespace

void cosmo_register_ui_routes(crow::SimpleApp &app) {
    CROW_ROUTE(app, "/v1/sdapi/v1/cosmopolitan-capabilities").methods("GET"_method)([] {
        crow::json::wvalue capabilities;
        capabilities["protocol"] = 1;
        capabilities["persistent_configuration"] = true;
        capabilities["configuration_schema"] = 1;
        capabilities["mode"] = "native-single-user";
        capabilities["kiosk_supported"] = false;
        capabilities["kiosk_enabled"] = false;
        capabilities["txt2img"] = true;
        capabilities["img2img_ui"] = false;
        capabilities["full_checkpoint_required"] = true;
        capabilities["companion_models_ui"] = false;
        capabilities["generation_plugins_ui"] = false;
        capabilities["output_formats"] = std::vector<std::string>{"png"};
        capabilities["max_concurrent_generations"] = 1;
        capabilities["cancel_during_model_load"] = false;
        crow::response response(capabilities);
        response.set_header("Cache-Control", "no-store");
        return response;
    });
    CROW_ROUTE(app, "/kiosk").methods("GET"_method)([] {
        crow::json::wvalue policy;
        policy["mode"] = "native-single-user";
        policy["supported"] = false;
        policy["enabled"] = false;
        policy["allowed_models"] = std::vector<std::string>{};
        crow::response response(policy);
        response.set_header("Cache-Control", "no-store");
        return response;
    });
    CROW_ROUTE(app, "/kiosk").methods("POST"_method)([] {
        crow::json::wvalue error;
        error["message"] = "Kiosk configuration is unavailable in this native build. No setting was changed.";
        return crow::response(501, error);
    });
    CROW_ROUTE(app, "/")([] { return page("/"); });
    // Crow reserves the slashless path as a redirect for a trailing-slash route.
    CROW_ROUTE(app, "/cpp-ui/")([] { return page("/"); });
    for (const auto &definition : easy_diffusion::ui::pages::definitions()) {
        if (definition.path == "/") continue;
        const std::string path = definition.path;
        app.route_dynamic("/cpp-ui" + path)([path] { return page(path); });
    }
    CROW_ROUTE(app, "/cpp-ui/assets/<path>")([](std::string path) { return resource("cpp-ui/assets/", path); });
    CROW_ROUTE(app, "/cpp-ui/scripts/<path>")([](std::string path) { return resource("cpp-ui/scripts/", path); });
    CROW_ROUTE(app, "/media/<path>")([](std::string path) { return resource("media/", path); });
    CROW_ROUTE(app, "/plugins/<path>")([](std::string path) { return resource("plugins/", path); });
}
