#include "inference/plan.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    using namespace edcpp::inference;

    GenerationRequest text_to_image;
    text_to_image.model = 41;
    text_to_image.prompt = "test prompt";
    const auto basic = make_plan(text_to_image);
    assert(basic.stages.size() == 6);
    assert(basic.stages.front() == Stage::tokenize);
    assert(basic.stages.back() == Stage::assemble_result);

    auto image_to_image = text_to_image;
    image_to_image.input = 9;
    const auto with_input = make_plan(image_to_image);
    assert(with_input.stages.size() == 7);
    assert(with_input.stages[2] == Stage::prepare_input);

    auto invalid = text_to_image;
    invalid.model = 0;
    assert(!validate(invalid).ok);
    bool threw = false;
    try { (void)make_plan(invalid); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}
