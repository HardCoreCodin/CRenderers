cmake_minimum_required (VERSION 3.8)

project(Win32Renderer)

set(SourceDir ${CMAKE_CURRENT_LIST_DIR}/src/)
set(BuildDir ${CMAKE_CURRENT_LIST_DIR}/bin/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BuildDir}) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BuildDir})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BuildDir})

include_directories(${SourceDir})

add_executable(Renderer WIN32 ${SourceDir}win32.c src/lib/core/memory.h src/lib/render/buffers.h src/lib/controls/fps.h src/lib/controls/orbit.h src/lib/controls/controls.h src/lib/nodes/camera.h src/lib/nodes/transform.h src/lib/render/engines/ray_tracer/shaders/ray_generation/ray_generator.h src/lib/render/engines/ray_tracer/engine.h src/lib/render/engines/ray_tracer/shaders/intersection/ray_sphere.h src/lib/render/engines/ray_tracer/ray.h src/lib/render/engines/ray_tracer/scene.h src/lib/render/engines/ray_tracer/shaders/closest_hit/normal.h)
set_target_properties(Renderer PROPERTIES OUTPUT_NAME win32)
target_link_libraries(Renderer PRIVATE kernel32.lib user32.lib gdi32.lib winmm.lib)
target_compile_definitions(Renderer PRIVATE PERF)

if (MSVC)
    add_compile_options(
            $<$<CONFIG:RELEASE>:-O2>
            $<$<CONFIG:RELEASE>:-Oi>
            $<$<CONFIG:RELEASE>:-Mt>
            $<$<CONFIG:DEBUG>:-JMC>
            $<$<CONFIG:DEBUG>:-MTd>
            $<$<CONFIG:DEBUG>:-Z7>
            -WX-
            -W4
            -wd4100
            -GR-
            -EHa-
            -GL
            -GS
            -FC
            -permissive-
            -sdl
            -nologo
            -analyze-
            -fp:fast
            -Zc:inline
            -diagnostics:caret
            -errorReport:none
    )
endif()