#include "opengl_backend.h"

#if defined(SDKIT_ENABLE_OPENGL_BACKEND)
#  if defined(_WIN32)
#    include <windows.h>
#  endif
#  include <GL/gl.h>
#endif

namespace sdkit::opengl {

bool compiled() {
#if defined(SDKIT_ENABLE_OPENGL_BACKEND)
    return true;
#else
    return false;
#endif
}

std::string implementation_name() {
    return "OpenGL/GLSL experimental backend";
}

std::string api_version() {
#if defined(SDKIT_ENABLE_OPENGL_BACKEND)
#  if defined(GL_VERSION_4_6)
    return "OpenGL headers >= 4.6";
#  elif defined(GL_VERSION_4_5)
    return "OpenGL headers >= 4.5";
#  elif defined(GL_VERSION_4_3)
    return "OpenGL headers >= 4.3";
#  else
    return "OpenGL headers available";
#  endif
#else
    return "OpenGL backend not compiled";
#endif
}

bool compute_ready() {
    // The first Theory milestone is compile/link plumbing only. This remains
    // false until SSBO allocation, GLSL compute kernels and GGML registration
    // are implemented.
    return false;
}

} // namespace sdkit::opengl
