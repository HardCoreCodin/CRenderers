#pragma once

#include "lib/core/types.h"
#include "lib/core/memory.h"
#include "lib/controls/fps.h"
#include "lib/controls/orbit.h"
#include "lib/controls/controls.h"

typedef struct EngineCore3D {
    Camera3D camera;
    FpsController3D fps_controller;
    OrbitController3D orbit_controller;
} EngineCore3D;

void initEngineCore3D(EngineCore3D* engine, Memory* memory) {
    initCamera3D(&engine->camera, memory);
    initFpsController(&engine->fps_controller);
    initOrbitController(&engine->orbit_controller, memory);
}

void onMousePositionChanged3D(f32 dx, f32 dy, Mouse* mouse, Buttons* buttons, EngineCore3D* engine) {
    if (mouse->is_captured)
        onOrient(-dx, -dy, &engine->fps_controller.orientation);
    else if (mouse->pressed) {
        if (mouse->pressed & buttons->MIDDLE)
            onPan(-dx, dy, &engine->orbit_controller.pan);
        else if (mouse->pressed & buttons->RIGHT)
            onOrbit(-dx, -dy, &engine->orbit_controller.orbit);
    }
}

void onMouseWheelChanged3D(f32 amount, Mouse* mouse, EngineCore3D* engine) {
    if (mouse->is_captured)
        onZoom(amount, &engine->fps_controller.zoom);
    else
        onDolly(amount, &engine->orbit_controller.dolly);
}