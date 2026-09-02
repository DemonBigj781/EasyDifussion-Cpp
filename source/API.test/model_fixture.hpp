#pragma once

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace api_test {

struct ModelFixture {
    std::string path;
    std::vector<unsigned char> bytes;
};

inline ModelFixture read_model_fixture(int argc, char** argv) {
    if (argc > 2) {
        throw std::runtime_error("usage: test-app [model-fixture]");
    }

    ModelFixture fixture;
    fixture.path = argc == 2
        ? argv[1]
        : "MODELS/lifecycle-model.fixture";

    std::ifstream input(fixture.path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "could not open model fixture: " + fixture.path);
    }

    fixture.bytes.assign(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
    if (fixture.bytes.empty()) {
        throw std::runtime_error(
            "model fixture is empty: " + fixture.path);
    }

    return fixture;
}

} // namespace api_test
