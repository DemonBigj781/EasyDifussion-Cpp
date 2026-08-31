#pragma once

#include <string>

namespace sdkit::opengl {

// Theory-stage OpenGL backend scaffold. This intentionally exposes compile-time
// capability only; GGML tensor execution will be added as GLSL kernels land.
bool compiled();
std::string implementation_name();
std::string api_version();
bool compute_ready();

} // namespace sdkit::opengl
