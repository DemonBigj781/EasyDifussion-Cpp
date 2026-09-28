#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace easy_diffusion::ui {

struct Attribute {
    std::string name;
    std::string value;
};

class Node {
public:
    static Node text(std::string value);
    static Node element(std::string tag,
                        std::vector<Attribute> attributes = {},
                        std::vector<Node> children = {});

    std::string render() const;

private:
    bool is_text_ = false;
    std::string value_;
    std::vector<Attribute> attributes_;
    std::vector<Node> children_;
};

class Page {
public:
    explicit Page(std::string title);

    Page& set_language(std::string language);
    Page& add_stylesheet(std::string url);
    Page& add_script(std::string url);
    Page& set_body(Node body);

    std::string render() const;

private:
    std::string title_;
    std::string language_ = "en";
    std::vector<std::string> stylesheets_;
    std::vector<std::string> scripts_;
    Node body_ = Node::element("main");
};

using Properties = std::map<std::string, std::string>;
using ComponentFactory = std::function<Node(const Properties&)>;

class ComponentRegistry {
public:
    void register_component(std::string name, ComponentFactory factory);
    Node render_component(const std::string& name, const Properties& properties = {}) const;
    bool contains(const std::string& name) const;

private:
    std::map<std::string, ComponentFactory> factories_;
};

}  // namespace easy_diffusion::ui
