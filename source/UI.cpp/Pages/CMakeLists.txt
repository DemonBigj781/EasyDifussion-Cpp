target_sources(easy-diffusion-ui PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pages.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Gallery/Image_Gallery.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Gallery/Window.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Gallery/Options.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Gallery/Navigate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Gallery/Load.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Settings/native_device_routing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/plugin_tab_inject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/controlnet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/display.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/image_options.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/image_view.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/lora.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/options.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/output_options.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/Main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/prompts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Generate/render_options.cpp
)
target_include_directories(easy-diffusion-ui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
