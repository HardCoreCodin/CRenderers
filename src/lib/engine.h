#pragma once

#include "lib/core/types.h"
#include "lib/core/hud.h"
#include "lib/core/perf.h"
#include "lib/core/text.h"
#include "lib/core/inputs.h"
#include "lib/nodes/scene.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"
#include "lib/input/controllers.h"
#include "lib/render/raytracing/raytracer.h"
#include "lib/render/raycasting/raycaster.h"

typedef enum Renderer {
    RAY_TRACE = 1,
    RAY_CAST = 2
} Renderer;
Renderer renderer = RAY_TRACE;

char* getEngineTitle() {
    return renderer == RAY_TRACE ? RAY_TRACER_TITLE : RAY_CASTER_TITLE;
}

void initRenderEngine() {
    initHUD();
    initInput();
    initFrameBuffer();

    initRayTracer();

    initScene();
}

void render() {

    rayTrace();
}

void OnFrameBufferResized() {
    updateHUDDimensions();
    if (renderer == RAY_TRACE)
        onResizeRT();
    else
        onResizeRC();
}

bool OnMouseDoubleClicked() {
    input.mouse.is_captured = !input.mouse.is_captured;
    setControllerModeInHUD(input.mouse.is_captured);
    return input.mouse.is_captured;
}

void OnMousePositionChanged(f32 dx, f32 dy) {
    if (input.mouse.is_captured)
        onOrient(-dx, -dy);
    else if (input.mouse.pressed) {
        if (input.mouse.pressed & input.buttons.MIDDLE)
            onPan(-dx, dy);
        else if (input.mouse.pressed & input.buttons.RIGHT)
            onOrbit(-dx, -dy);
    }
}

void OnMouseWheelChanged(f32 amount) {
    if (input.mouse.is_captured)
        onZoom(amount);
    else
        onDolly(amount);
}

void update2D(f32 delta_time, Transform2D* transform, f32* focal_length) {
    if (input.mouse.is_captured) {

        if (zoom.changed) {
            updateZoom(focal_length);
            if (renderer == RAY_TRACE) onZoomRC();
        }

        if (orientation.changed) {
            updateOrientation2D(transform);
            if (renderer == RAY_TRACE) onOrientRC();
        }
    } else {
        if (orbit.changed) {
            updateOrbit2D(transform);
            if (renderer == RAY_TRACE) onOrbitRC();
        }
        if (pan.changed)
            updatePan2D(transform);
    }

    onMove2D(transform, delta_time);
}

void update3D(f32 delta_time, Transform3D* transform, f32* focal_length) {
    if (input.mouse.is_captured) {

        if (zoom.changed) {
            updateZoom(focal_length);
            if (renderer == RAY_TRACE) onZoomRT();
        }

        if (orientation.changed) {
            updateOrientation3D(transform);
            if (renderer == RAY_TRACE) onOrientRT();
        }
    } else {
        if (orbit.changed) {
            updateOrbit3D(transform);
            if (renderer == RAY_TRACE) onOrbitRT();
        }
        if (pan.changed)
            updatePan3D(transform);

        if (dolly.changed)
            updateDolly(transform);
    }

    onMove3D(transform, delta_time);
}

void update() {
    f32 delta_time = (f32)hud.main_perf.delta.seconds;
    if (delta_time > 1)
        delta_time = 1;

    if (input.keyboard.pressed & input.buttons.HUD) {
        input.keyboard.pressed &= (u8)~input.buttons.HUD;
        hud.is_visible = !hud.is_visible;
    }

    if (renderer == RAY_TRACE)
        update3D(delta_time, ray_tracer.camera.transform, &ray_tracer.camera.focal_length);
    else if (renderer == RAY_CAST)
        update2D(delta_time, ray_caster.camera.transform, &ray_caster.camera.focal_length);
}

void OnFrameUpdate() {
    PERF_START_FRAME(hud.main_perf)

    update();
    render();

    PERF_FRAME_END(hud.main_perf)
    if (hud.is_visible) {
        if (!hud.main_perf.accum.frames)
            updateHUDCounters();
        drawText(hud.text, HUD_COLOR, frame_buffer.width - HUD_RIGHT - HUD_WIDTH, HUD_TOP);
    }
}