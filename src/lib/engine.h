#pragma once

#include "lib/core/types.h"
#include "lib/core/hud.h"
#include "lib/core/perf.h"
#include "lib/core/text.h"
#include "lib/input/mouse.h"
#include "lib/input/keyboard.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/nodes/scene.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"
#include "lib/engine.h"
#include "lib/render/raytracing/raytracer.h"
#include "lib/render/raycasting/raycaster.h"

static Engine engine;

void onUpdateAndRender() {
    PERF_START_FRAME

    Controller* controller = engine.renderer->controller;

    if (mouse.wheel.was_scrolled) {
        mouse.wheel.was_scrolled = false;
        controller->on.mouse_scrolled();
        mouse.wheel.scroll_amount = 0;
    }

    if (mouse.has_moved) {
        mouse.has_moved = false;
        controller->on.mouse_moved();
        mouse.coords.relative.x = 0;
        mouse.coords.relative.y = 0;
    }

    if (buttons.hud.is_pressed) {
        buttons.hud.is_pressed = false;
        hud.is_visible = !hud.is_visible;
    }

    if (mouse.double_clicked) {
        mouse.double_clicked = false;
        engine.renderer->on.double_clicked();
    }

    controller->on.update();

    engine.renderer->on.render();

    PERF_END_FRAME
    if (hud.is_visible) {
        if (!perf.accum.frames)
            updateHUDCounters();
        drawText(hud.text, HUD_COLOR, frame_buffer.width - HUD_RIGHT - HUD_WIDTH, HUD_TOP);
    }

    if (buttons.first.is_pressed) engine.renderer = &ray_tracer.base;
    if (buttons.second.is_pressed) engine.renderer = &ray_caster.base;
}

void initEngine() {
    engine.is_running = true;
    engine.in_fps_mode = false;

    initPerf(&perf);

    initHUD();
    initMouse();
    initButtons();

    initScene();
    initFrameBuffer();
    initRayTracer();
    initRayCaster();

    engine.renderer = &ray_tracer.base;
    engine.updateAndRender = onUpdateAndRender;
}