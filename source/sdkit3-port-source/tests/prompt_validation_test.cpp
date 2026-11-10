#include "prompt_validation.h"
#include "native_prompt_validation.h"
#include <iostream>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
static std::string validate(const std::string& prompt, const char* field = "prompt") {
    crow::json::wvalue request;
    request[field] = prompt;
    const auto serialized = request.dump();
    const auto parsed = crow::json::load(serialized);
    const auto result = validate_generation_prompts(parsed);
    require(std::string(parsed[field].s()) == prompt, "validation changed prompt bytes");
    return result;
}
int main() {
    try {
        require(sdkit_validate_prompt_attention(nullptr, 0) == SDKIT_PROMPT_VALIDATION_FAILED,
                "null adapter input must not escape or succeed");
        const std::string binary("prefix\0(invalid:.)", 18);
        require(!validate(binary).empty(), "adapter lost bytes after embedded NUL");
        for (const auto& prompt : std::vector<std::string>{
                 "", "normal text", "an (important) word", "[quiet]", "(unbalanced",
                 "(half:+.5) (none:0) (negative:-1)", "one BREAK two", "red:fur AND green:eyes",
                 "xyz++", "literal:99999999999999999999999999999999999999999999999999",
                 "\\(literal\\]", "日本語 (猫:1.2)", "(outer [inner])"}) {
            require(validate(prompt).empty(), "rejected compatible prompt");
            require(validate(prompt, "negative_prompt").empty(), "rejected compatible negative prompt");
        }
        for (const auto& prompt : std::vector<std::string>{
                 "(invalid:.)", "(invalid:+.)",
                 "(overflow:99999999999999999999999999999999999999999999999999)",
                 "((overflow:100000000000000000000):100000000000000000000)"}) {
            require(validate(prompt).find("prompt ") == 0, "invalid prompt was accepted");
            require(validate(prompt, "negative_prompt").find("negative_prompt ") == 0,
                    "invalid negative prompt was accepted");
        }
        require(validate_generation_prompts(crow::json::load("{}")).empty(), "missing prompts must stay valid");
        for (const auto* json : {"[]", "null", "{\"prompt\":42}", "{\"negative_prompt\":null}"}) {
            require(!validate_generation_prompts(crow::json::load(json)).empty(), "invalid request type accepted");
        }
        std::cout << "Prompt preflight: actual parser, numeric failures, negative fields and unchanged text passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
