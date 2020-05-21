cmake_minimum_required (VERSION 3.8)

project(Win32Renderer)

set(SourceDir ${CMAKE_CURRENT_LIST_DIR}/src/)
set(BuildDir ${CMAKE_CURRENT_LIST_DIR}/bin/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BuildDir}) 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BuildDir})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BuildDir})

include_directories(${SourceDir})

add_executable(Renderer WIN32 ${SourceDir}win32.c)
set_target_properties(Renderer PROPERTIES OUTPUT_NAME win32)
target_link_libraries(Renderer PRIVATE kernel32.lib user32.lib gdi32.lib winmm.lib)
target_compile_definitions(Renderer PRIVATE PERF)

if (MSVC)
    add_compile_options(
            $<$<CONFIG:RELEASE>:-O3>
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