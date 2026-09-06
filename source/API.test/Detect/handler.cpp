#include EDCPP_HANDLER_HEADER

#if EDCPP_DETECT_CREATE_EGL_CONTEXT
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#include <cstdlib>
#include <iostream>
#include <set>
#include <string_view>

namespace {

#if EDCPP_DETECT_CREATE_EGL_CONTEXT
class EglContext {
public:
    bool create() {
        const auto get_platform_display =
            reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
                eglGetProcAddress("eglGetPlatformDisplayEXT"));
        if (get_platform_display == nullptr) return false;

        display_ = get_platform_display(
            EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        if (display_ == EGL_NO_DISPLAY ||
            !eglInitialize(display_, nullptr, nullptr) ||
            !eglBindAPI(EGL_OPENGL_API)) {
            return false;
        }

        const EGLint config_attributes[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_NONE,
        };
        EGLConfig config = nullptr;
        EGLint config_count = 0;
        if (!eglChooseConfig(
                display_, config_attributes, &config, 1, &config_count) ||
            config_count == 0) {
            return false;
        }

        const EGLint surface_attributes[] = {
            EGL_WIDTH, 1,
            EGL_HEIGHT, 1,
            EGL_NONE,
        };
        surface_ = eglCreatePbufferSurface(
            display_, config, surface_attributes);
        context_ = eglCreateContext(
            display_, config, EGL_NO_CONTEXT, nullptr);
        return surface_ != EGL_NO_SURFACE && context_ != EGL_NO_CONTEXT &&
               eglMakeCurrent(display_, surface_, surface_, context_);
    }

    ~EglContext() {
        if (display_ == EGL_NO_DISPLAY) return;
        eglMakeCurrent(
            display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
        if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
        eglTerminate(display_);
    }

private:
    EGLDisplay display_ = EGL_NO_DISPLAY;
    EGLSurface surface_ = EGL_NO_SURFACE;
    EGLContext context_ = EGL_NO_CONTEXT;
};
#endif

} // namespace

int main() {
#if EDCPP_DETECT_CREATE_EGL_CONTEXT
    EglContext context;
    if (!context.create()) {
        std::cerr << EDCPP_HANDLER_NAME
                  << " detect test could not create a surfaceless EGL context\n";
        return EXIT_FAILURE;
    }
#endif

    EDCPP_HANDLER_TYPE handler;
    if (std::string_view(handler.name()) != EDCPP_HANDLER_NAME) {
        std::cerr << "detect handler returned the wrong backend name\n";
        return EXIT_FAILURE;
    }
    if (!handler.available()) {
        std::cerr << EDCPP_HANDLER_NAME << " backend is unavailable\n";
        return EXIT_FAILURE;
    }

    const auto devices = handler.devices();
    if (devices.empty()) {
        std::cerr << EDCPP_HANDLER_NAME
                  << " backend reported available without a device\n";
        return EXIT_FAILURE;
    }
    std::set<int> indices;
    for (const auto& device : devices) {
        if (!device.available || device.name.empty() ||
            device.backend != EDCPP_HANDLER_NAME) {
            std::cerr << EDCPP_HANDLER_NAME
                      << " detector returned an invalid device record\n";
            return EXIT_FAILURE;
        }
        if (!indices.insert(device.index).second) {
            std::cerr << EDCPP_HANDLER_NAME
                      << " detector returned a duplicate device index\n";
            return EXIT_FAILURE;
        }
        if (device.total_memory != 0 &&
            device.free_memory > device.total_memory) {
            std::cerr << EDCPP_HANDLER_NAME
                      << " detector reported free memory above total memory\n";
            return EXIT_FAILURE;
        }
        std::cout << device.backend << '[' << device.index << "] "
                  << device.name << " arch=" << device.architecture
                  << " memory=" << (device.free_memory / (1024 * 1024)) << '/'
                  << (device.total_memory / (1024 * 1024)) << " MiB\n";
    }
    return EXIT_SUCCESS;
}
