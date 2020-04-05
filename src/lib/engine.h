#pragma once

#include "lib/core/types.h"
#include "lib/core/hud.h"
#include "lib/core/perf.h"
#include "lib/core/text.h"
#include "lib/input/mouse.h"
#include "lib/input/keyboard.h"
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

typedef struct RenderEngine {
    Renderer renderer;
    bool is_running, in_fps_mode;
} RenderEngine;
RenderEngine engine;

char* getEngineTitle() {
    return engine.renderer == RAY_TRACE ? RAY_TRACER_TITLE : RAY_CASTER_TITLE;
}

void initRenderEngine() {
    initPerf(&perf);

    initHUD();
    initMouse();
    initButtons();

    initFrameBuffer();
    initRayTracer();
    initScene();

    engine.is_running = true;
    engine.in_fps_mode = false;
    engine.renderer = RAY_TRACE;
}

void render() {
    switch (engine.renderer) {
        case RAY_TRACE: rayTrace(); break;
        case RAY_CAST: rayCast(); break;
    }
}

void OnFrameBufferResized() {
    updateHUDDimensions();
    switch (engine.renderer) {
        case RAY_TRACE: onResizeRT(); break;
        case RAY_CAST: onResizeRC(); break;
    }
}

void update() {
    if (mouse.double_click.was_actioned) {
        mouse.double_click.was_actioned = false;
        engine.in_fps_mode = !engine.in_fps_mode;
        setControllerModeInHUD(engine.in_fps_mode);
    }

    if (mouse.wheel.was_scrolled) {
        mouse.wheel.was_scrolled = false;

        if (engine.in_fps_mode)
            onZoom(mouse.wheel.scroll_amount);
        else
            onDolly(mouse.wheel.scroll_amount);

        mouse.wheel.scroll_amount = 0;
    }

    if (mouse.has_moved) {
        mouse.has_moved = false;

        if (engine.in_fps_mode)
            onOrient(-mouse.coords.relative.x, -mouse.coords.relative.y);
        else if (mouse.buttons.middle.is_down)
            onPan(-mouse.coords.relative.x, mouse.coords.relative.y);
        else if (mouse.buttons.right.is_down)
            onOrbit(-mouse.coords.relative.x, -mouse.coords.relative.y);

        mouse.coords.relative.x = 0;
        mouse.coords.relative.y = 0;
    }

    if (buttons.hud.is_pressed) {
        buttons.hud.is_pressed = false;
        hud.is_visible = !hud.is_visible;
    }

    f32 delta_time = (f32)perf.delta.seconds;
    if (delta_time > 1)
        delta_time = 1;

    if (engine.in_fps_mode) {
        if (zoom.changed) {
            switch (engine.renderer) {
                case RAY_TRACE: onZoomRT(); updateZoom(&ray_tracer.camera.focal_length); break;
                case RAY_CAST: onZoomRC(); updateZoom(&ray_caster.camera.focal_length); break;
            }
        }

        if (orientation.changed) {
            switch (engine.renderer) {
                case RAY_TRACE: updateOrientation3D(ray_tracer.camera.transform); onOrientRT(); break;
                case RAY_CAST: updateOrientation2D(ray_caster.camera.transform); onOrientRC(); break;
            }
        }
    } else {
        if (orbit.changed) {
            switch (engine.renderer) {
                case RAY_TRACE: updateOrbit3D(ray_tracer.camera.transform); onOrbitRT(); break;
                case RAY_CAST: updateOrbit2D(ray_caster.camera.transform); onOrbitRC(); break;
            }
        }
        if (pan.changed)
            switch (engine.renderer) {
                case RAY_TRACE: updatePan3D(ray_tracer.camera.transform); break;
                case RAY_CAST: updatePan2D(ray_caster.camera.transform); break;
            }

        if (dolly.changed)
            switch (engine.renderer) {
                case RAY_TRACE: updateDolly(ray_tracer.camera.transform); break;
                case RAY_CAST: break;
            }
    }

    switch (engine.renderer) {
        case RAY_TRACE: onMove3D(ray_tracer.camera.transform->yaw, ray_tracer.camera.transform->position, delta_time); break;
        case RAY_CAST: onMove2D(ray_caster.camera.transform->rotation, ray_caster.camera.transform->position, delta_time); break;
    }
}

void OnFrameUpdate() {
    PERF_START_FRAME

    update();
    render();

    PERF_END_FRAME
    if (hud.is_visible) {
        if (!perf.accum.frames)
            updateHUDCounters();
        drawText(hud.text, HUD_COLOR, frame_buffer.width - HUD_RIGHT - HUD_WIDTH, HUD_TOP);
    }
}