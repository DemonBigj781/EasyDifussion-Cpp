# Generic defaults to .obj, but the fat Cosmopolitan driver expects .o when
# classifying linker inputs. Load this after CMake's language defaults, also
# for try_compile(), so object naming never depends on the build host's OS.
set(CMAKE_C_OUTPUT_EXTENSION .o)
set(CMAKE_CXX_OUTPUT_EXTENSION .o)
