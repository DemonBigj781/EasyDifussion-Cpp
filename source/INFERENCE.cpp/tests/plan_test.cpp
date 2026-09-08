#include "inference/plan.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    using namespace edcpp::inference;

    edcpp::api::load::Resource model;
    model.backend = edcpp::api::Backend::cpu;
    model.native_handle = reinterpret_cast<void*>(1);
    model.size = 1;
    GenerationRequest text_to_image;
    text_to_image.model = &model;
    text_to_image.prompt = "test prompt";
    const auto basic = make_plan(text_to_image);
    assert(basic.stages.size() == 6);
    assert(basic.stages.front() == Stage::tokenize);
    assert(basic.stages.back() == Stage::assemble_result);

    auto invalid = text_to_image;
    invalid.model = nullptr;
    assert(!validate(invalid).ok);
    bool threw = false;
    try { (void)make_plan(invalid); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}
