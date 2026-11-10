#include "features/conditioning/common/prompt_attention.hpp"

#include <cassert>
#include <cmath>
#include <string>

namespace {

bool near(float left, float right) {
    return std::fabs(left - right) < 0.0001f;
}

} // namespace

int main() {
    using namespace edcpp::inference;

    const auto plain = parse_prompt_attention("normal text");
    assert(plain.ok && plain.segments.size() == 1);
    assert(plain.segments[0].text == "normal text");
    assert(near(plain.segments[0].weight, 1.0f));

    const auto emphasized = parse_prompt_attention("an (important) word");
    assert(emphasized.ok && emphasized.segments.size() == 3);
    assert(emphasized.segments[0].text == "an ");
    assert(emphasized.segments[1].text == "important");
    assert(emphasized.segments[2].text == " word");
    assert(near(emphasized.segments[1].weight, 1.1f));

    const auto unbalanced = parse_prompt_attention("(unbalanced");
    assert(unbalanced.ok && unbalanced.segments.size() == 1);
    assert(unbalanced.segments[0].text == "unbalanced");
    assert(near(unbalanced.segments[0].weight, 1.1f));

    const auto escaped = parse_prompt_attention("\\(literal\\] \\\\");
    assert(escaped.ok && escaped.segments.size() == 1);
    assert(escaped.segments[0].text == "(literal] \\");
    assert(near(escaped.segments[0].weight, 1.0f));

    const auto merged = parse_prompt_attention("(unnecessary)(parens)");
    assert(merged.ok && merged.segments.size() == 1);
    assert(merged.segments[0].text == "unnecessaryparens");
    assert(near(merged.segments[0].weight, 1.1f));

    const auto explicit_weight =
        parse_prompt_attention("(half:+.5) and (none:0)");
    assert(explicit_weight.ok && explicit_weight.segments.size() == 3);
    assert(explicit_weight.segments[0].text == "half");
    assert(near(explicit_weight.segments[0].weight, 0.5f));
    assert(explicit_weight.segments[1].text == " and ");
    assert(explicit_weight.segments[2].text == "none");
    assert(near(explicit_weight.segments[2].weight, 0.0f));

    const auto lowered = parse_prompt_attention("[quiet]");
    assert(lowered.ok && lowered.segments.size() == 1);
    assert(near(lowered.segments[0].weight, 1.0f / 1.1f));

    const auto nested = parse_prompt_attention("(outer [inner])");
    assert(nested.ok && nested.segments.size() == 2);
    assert(nested.segments[0].text == "outer ");
    assert(near(nested.segments[0].weight, 1.1f));
    assert(nested.segments[1].text == "inner");
    assert(near(nested.segments[1].weight, 1.0f));

    const auto separated = parse_prompt_attention("one  BREAK \t two");
    assert(separated.ok && separated.segments.size() == 3);
    assert(separated.segments[0].text == "one");
    assert(separated.segments[1].kind == PromptSegmentKind::break_marker);
    assert(separated.segments[2].text == "two");

    const auto weighted_break = parse_prompt_attention("(one BREAK two)");
    assert(weighted_break.ok && weighted_break.segments.size() == 3);
    assert(near(weighted_break.segments[0].weight, 1.1f));
    assert(weighted_break.segments[1].kind == PromptSegmentKind::break_marker);
    assert(near(weighted_break.segments[1].weight, 1.0f));
    assert(near(weighted_break.segments[2].weight, 1.1f));

    const auto break_word = parse_prompt_attention("BREAKFAST and tea");
    assert(break_word.ok && break_word.segments.size() == 1);
    assert(break_word.segments[0].text == "BREAKFAST and tea");

    const auto natural_relations =
        parse_prompt_attention("red:fur AND green:eyes xyz++");
    assert(natural_relations.ok && natural_relations.segments.size() == 2);
    assert(natural_relations.segments[0].text ==
           "red:fur AND green:eyes ");
    assert(near(natural_relations.segments[0].weight, 1.0f));
    assert(natural_relations.segments[1].text == "xyz");
    assert(near(natural_relations.segments[1].weight, 1.21f));

    const auto natural_bare_weight =
        parse_prompt_attention("xyz:1.2 at 16:9");
    assert(natural_bare_weight.ok && natural_bare_weight.segments.size() == 2);
    assert(natural_bare_weight.segments[0].text == "xyz");
    assert(near(natural_bare_weight.segments[0].weight, 1.2f));
    assert(natural_bare_weight.segments[1].text == " at 16:9");

    const auto tag_relations = parse_prompt_attention(
        "red:fur green:eyes red:orange:chest", PromptMode::tags);
    assert(tag_relations.ok && tag_relations.segments.size() == 5);
    assert(tag_relations.segments[0].text == "red fur");
    assert(tag_relations.segments[0].fusion_group != 0);
    assert(tag_relations.segments[2].text == "green eyes");
    assert(tag_relations.segments[2].fusion_group !=
           tag_relations.segments[0].fusion_group);
    assert(tag_relations.segments[4].text == "red orange chest");

    const auto censored_shape = parse_prompt_attention(
        "(white:color:subject), (white:color:other), (furry:female)",
        PromptMode::tags);
    assert(censored_shape.ok);
    assert(censored_shape.segments[0].text == "white color subject");
    assert(near(censored_shape.segments[0].weight, 1.1f));
    assert(censored_shape.segments[0].fusion_group != 0);

    const auto bare_weight =
        parse_prompt_attention("red:fur:1.2", PromptMode::tags);
    assert(bare_weight.ok && bare_weight.segments.size() == 1);
    assert(bare_weight.segments[0].text == "red fur");
    assert(near(bare_weight.segments[0].weight, 1.2f));
    assert(bare_weight.segments[0].fusion_group != 0);

    const auto plus_weight =
        parse_prompt_attention("xyz++", PromptMode::tags);
    assert(plus_weight.ok && plus_weight.segments.size() == 1);
    assert(plus_weight.segments[0].text == "xyz");
    assert(near(plus_weight.segments[0].weight, 1.21f));

    const auto conjunction =
        parse_prompt_attention("red:fur AND green:eyes", PromptMode::tags);
    assert(conjunction.ok && conjunction.segments.size() == 3);
    assert(conjunction.segments[1].kind ==
           PromptSegmentKind::conjunction_marker);

    const auto unmatched_closing = parse_prompt_attention("plain) text]");
    assert(unmatched_closing.ok && unmatched_closing.segments.size() == 1);
    assert(unmatched_closing.segments[0].text == "plain) text]");

    const auto malformed = parse_prompt_attention("(x:.)");
    assert(malformed.ok && malformed.segments.size() == 1);
    assert(malformed.segments[0].text == "x:.");
    assert(near(malformed.segments[0].weight, 1.1f));

    const auto overflow = parse_prompt_attention(
        "(x:999999999999999999999999999999999999999999999999999)");
    assert(!overflow.ok && !overflow.message.empty());

    const auto empty = parse_prompt_attention("");
    assert(empty.ok && empty.segments.size() == 1);
    assert(empty.segments[0].kind == PromptSegmentKind::text);
    assert(empty.segments[0].text.empty());
}
