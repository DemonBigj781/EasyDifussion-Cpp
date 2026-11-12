#include <cmath>
#include <iostream>
#include <stdexcept>

#include "core/util.h"
#include "native_prompt_validation.h"

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void expect_single(const std::string& prompt, const std::string& text, float weight) {
    const auto parsed = parse_prompt_attention(prompt);
    require(parsed.size() == 1 && parsed[0].first == text && std::abs(parsed[0].second - weight) < 0.00001f,
            "prompt text or attention weight changed");
}

int main() {
    try {
        const std::string long_text(65536, 'x');
        expect_single(long_text, long_text, 1.0f);
        require(sdkit_validate_prompt_attention(long_text.data(), long_text.size()) == SDKIT_PROMPT_VALID,
                "native admission rejected a long plain prompt");
        expect_single("(" + long_text + ":1.25)", long_text, 1.25f);
        std::string unicode;
        for (int i = 0; i < 12000; ++i) unicode += "猫";
        expect_single(unicode, unicode, 1.0f);
        expect_single("(long-weight:" + std::string(65536, '0') + "1)", "long-weight", 1.0f);
        expect_single("plain", "plain", 1.0f);
        expect_single("((nested))", "nested", 1.21f);
        expect_single("[quiet]", "quiet", 1.0f / 1.1f);
        expect_single("(half:+.5)", "half", 0.5f);
        expect_single("(negative:-1)", "negative", -1.0f);
        expect_single("(zero:0)", "zero", 0.0f);
        expect_single("(unbalanced", "unbalanced", 1.1f);
        expect_single("(legacy:1.2.3)", "legacy", 1.2f);
        expect_single("outside:.)", "outside", 1.0f);
        expect_single("\\(literal", "(literal", 1.0f);
        const auto split = parse_prompt_attention(long_text + " BREAK " + long_text);
        require(split.size() == 3 && split[1].first == "BREAK" && split[1].second == -1.0f,
                "BREAK boundary changed");
        for (const std::string& prompt : {std::string("(bad:.)"), std::string("(bad:+.)")}) {
            bool rejected = false;
            try { parse_prompt_attention(prompt); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid numeric weight no longer rejected");
        }
        bool overflow = false;
        try { parse_prompt_attention("(overflow:" + std::string(65536, '9') + ")"); }
        catch (const std::out_of_range&) { overflow = true; }
        require(overflow, "overflow no longer rejected");
        const std::string huge_weight = "(overflow:" + std::string(65536, '9') + ")";
        require(sdkit_validate_prompt_attention(huge_weight.data(), huge_weight.size()) == SDKIT_PROMPT_WEIGHT_OUT_OF_RANGE,
                "native admission lost numeric range rejection");
        const std::string invalid_weight = "(bad:.)";
        require(sdkit_validate_prompt_attention(invalid_weight.data(), invalid_weight.size()) == SDKIT_PROMPT_INVALID_WEIGHT,
                "native admission lost invalid weight rejection");
        std::cout << "PASS long plain/Unicode/numeric runs, attention semantics, BREAK and preserved numeric errors\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
