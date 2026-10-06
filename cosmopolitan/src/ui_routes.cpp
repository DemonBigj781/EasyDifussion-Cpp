#include "ui_routes.hpp"
#include "easy_diffusion/ui/pages.hpp"

#include <cstdio>
#include <string>

namespace {

crow::response page(const std::string &path) {
    try {
        crow::response result(easy_diffusion::ui::pages::render(path).render());
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
