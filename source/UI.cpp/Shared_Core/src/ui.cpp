#include "easy_diffusion/ui.hpp"

#include <algorithm>
#include <cctype>
#include <set>
#include <stdexcept>
#include <utility>

namespace easy_diffusion::ui {
namespace {

std::string escape_html(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            case '\'': escaped += "&#39;"; break;
            default: escaped += c;
        }
    }
    return escaped;
}

bool valid_name(const std::string& value) {
    if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value.front())) || value.front() == '_'))
        return false;
    return std::all_of(value.begin() + 1, value.end(), [](unsigned char c) {
        return std::isalnum(c) || c == '-' || c == '_' || c == ':' || c == '.';
    });
}

bool is_void_element(const std::string& tag) {
    static const std::set<std::string> tags = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    };
    return tags.find(tag) != tags.end();
}

bool safe_asset_url(const std::string& url) {
    if (url.empty() || url.find_first_of("\r\n\t\\") != std::string::npos || url.rfind("//", 0) == 0)
        return false;
    const auto colon = url.find(':');
    const auto slash = url.find('/');
    const auto query = url.find('?');
    const auto fragment = url.find('#');
    const auto boundary = std::min({slash == std::string::npos ? url.size() : slash,
                                    query == std::string::npos ? url.size() : query,
                                    fragment == std::string::npos ? url.size() : fragment});
    if (colon == std::string::npos || colon > boundary) return true;
    std::string scheme = url.substr(0, colon);
    std::transform(scheme.begin(), scheme.end(), scheme.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return scheme == "http" || scheme == "https";
}

}  // namespace

Node Node::text(std::string value) {
    Node node;
    node.is_text_ = true;
    node.value_ = std::move(value);
    return node;
}

Node Node::element(std::string tag, std::vector<Attribute> attributes, std::vector<Node> children) {
    std::string lower_tag = tag;
    std::transform(lower_tag.begin(), lower_tag.end(), lower_tag.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (!valid_name(tag) || lower_tag == "script" || lower_tag == "style")
        throw std::invalid_argument("Invalid or inline-active HTML element");
    for (const auto& attribute : attributes) {
        if (!valid_name(attribute.name)) throw std::invalid_argument("Invalid HTML attribute name");
        std::string lower = attribute.name;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (lower.rfind("on", 0) == 0) throw std::invalid_argument("Inline event handlers are not allowed");
        if ((lower == "href" || lower == "src" || lower == "action" || lower == "formaction") &&
            !safe_asset_url(attribute.value)) {
            throw std::invalid_argument("Unsafe URL in HTML attribute");
        }
    }
    if (is_void_element(lower_tag) && !children.empty())
        throw std::invalid_argument("Void HTML elements cannot have children");
    Node node;
    node.value_ = std::move(tag);
    node.attributes_ = std::move(attributes);
    node.children_ = std::move(children);
    return node;
}

std::string Node::render() const {
    if (is_text_) return escape_html(value_);
    std::string html = "<" + value_;
    for (const auto& attribute : attributes_)
        html += " " + attribute.name + "=\"" + escape_html(attribute.value) + "\"";
    html += ">";
    if (is_void_element(value_)) return html;
    for (const auto& child : children_) html += child.render();
    return html + "</" + value_ + ">";
}

Page::Page(std::string title) : title_(std::move(title)) {}

Page& Page::set_language(std::string language) {
    if (!valid_name(language)) throw std::invalid_argument("Invalid page language");
    language_ = std::move(language);
    return *this;
}

Page& Page::add_stylesheet(std::string url) {
    if (!safe_asset_url(url)) throw std::invalid_argument("Unsafe stylesheet URL");
    stylesheets_.push_back(std::move(url));
    return *this;
}

Page& Page::add_script(std::string url) {
    if (!safe_asset_url(url)) throw std::invalid_argument("Unsafe script URL");
    scripts_.push_back(std::move(url));
    return *this;
}

Page& Page::set_body(Node body) {
    body_ = std::move(body);
    return *this;
}

std::string Page::render() const {
    std::string html = "<!doctype html><html lang=\"" + escape_html(language_) +
        "\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
        "content=\"width=device-width, initial-scale=1\"><title>" + escape_html(title_) + "</title>";
    for (const auto& url : stylesheets_)
        html += "<link rel=\"stylesheet\" href=\"" + escape_html(url) + "\">";
    html += "</head><body>" + body_.render();
    for (const auto& url : scripts_)
        html += "<script defer src=\"" + escape_html(url) + "\"></script>";
    return html + "</body></html>";
}

void ComponentRegistry::register_component(std::string name, ComponentFactory factory) {
    if (name.empty() || !factory) throw std::invalid_argument("Component name and factory are required");
    if (!factories_.emplace(std::move(name), std::move(factory)).second)
        throw std::invalid_argument("UI component is already registered");
}

Node ComponentRegistry::render_component(const std::string& name, const Properties& properties) const {
    const auto found = factories_.find(name);
    if (found == factories_.end()) throw std::out_of_range("Unknown UI component: " + name);
    return found->second(properties);
}

bool ComponentRegistry::contains(const std::string& name) const {
    return factories_.find(name) != factories_.end();
}

}  // namespace easy_diffusion::ui
