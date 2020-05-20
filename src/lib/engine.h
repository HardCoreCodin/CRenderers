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

char* getTitle() {
    return engine.viewport.renderer->title;
}

void resize() {
    switch (engine.viewport.renderer->type) {
        case RENDERER_RT: onResizeRT(&engine.viewport); break;
        case RENDERER_RC: onResizeRC(&engine.viewport); break;
    }
    updateHUDDimensions();
}

void toggleControllerMode() {
    engine.viewport.in_fps_mode = !engine.viewport.in_fps_mode;
    engine.viewport.controller = engine.viewport.in_fps_mode ? &fps.controller : &orb.controller;
    setControllerModeInHUD(engine.viewport.in_fps_mode);
}

void updateAndRender() {
    PERF_START_FRAME

    Controller* controller = engine.viewport.controller;

    if (mouse.wheel.changed) {
        mouse.wheel.changed = false;
        switch (controller->type) {
            case CONTROLLER_FPS: onMouseScrolledFps(); break;
            case CONTROLLER_ORB: onMouseScrolledOrb(); break;
        }
        mouse.wheel.scroll = 0;
    }

    if (mouse.coords.relative.changed) {
        mouse.coords.relative.changed = false;
        switch (controller->type) {
            case CONTROLLER_FPS: onMouseMovedFps(); break;
            case CONTROLLER_ORB: onMouseMovedOrb(); break;
        }
        mouse.coords.relative.x = 0;
        mouse.coords.relative.y = 0;
    }

    switch (controller->type) {
        case CONTROLLER_FPS:
            onUpdateFps();
            break;
        case CONTROLLER_ORB:
            onUpdateOrb();
            break;
    }

    if (controller->changed.fov) {
        controller->changed.fov = false;
        switch (engine.viewport.renderer->type) {
            case RENDERER_RT: onZoomRT(&engine.viewport); break;
            case RENDERER_RC: onZoomRC(&engine.viewport); break;
        }
    }

    if (controller->changed.orientation) {
        controller->changed.orientation = false;
        switch (engine.viewport.renderer->type) {
            case RENDERER_RT: onRotateRT(&engine.viewport); break;
            case RENDERER_RC: onRotateRC(&engine.viewport); break;
        }
    }

    if (controller->changed.position) {
        controller->changed.position = false;
        switch (engine.viewport.renderer->type) {
            case RENDERER_RT: onMoveRT(&engine.viewport); break;
            case RENDERER_RC: onMoveRC(&engine.viewport); break;
        }
    }

    switch (engine.viewport.renderer->type) {
        case RENDERER_RT: onRenderRT(&engine.viewport); break;
        case RENDERER_RC: onRenderRC(&engine.viewport); break;
    }

    PERF_END_FRAME
    if (hud.is_visible) {
        if (!perf.accum.frames)
            updateHUDCounters();
        drawText(hud.text, HUD_COLOR, frame_buffer.width - HUD_RIGHT - HUD_WIDTH, HUD_TOP);
    }

//    if (buttons.first.is_pressed) engine.renderer = &ray_tracer.renderer;
//    if (buttons.second.is_pressed) engine.renderer = &ray_caster.base;


    if (buttons.hud.is_pressed) {
        buttons.hud.is_pressed = false;
        hud.is_visible = !hud.is_visible;
    }

    if (mouse.double_clicked) {
        mouse.double_clicked = false;
        toggleControllerMode(engine.viewport);
    }
}

void initEngine(Callback updateWindowTitle) {
    initPerf(&perf);

    initHUD();
    initMouse();
    initButtons();

    initFrameBuffer();
    initScene(&engine.scene);
    initRayTracer(&engine.scene);
    initRayCaster(&engine.scene);

    orb.controller.camera = engine.scene.camera;
    fps.controller.camera = engine.scene.camera;

    engine.is_running = true;
    engine.viewport.controller = &orb.controller;
    engine.viewport.renderer = &ray_tracer.renderer;
    engine.scene.camera->transform->position->x = 5;
    engine.scene.camera->transform->position->y = 5;
    engine.scene.camera->transform->position->z = -10;
    orb.controller.changed.position = true;
}