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

char* getTitle(Engine* engine) {
    return engine->active_viewport->renderer->title;
}

void resize(Engine* engine) {
    switch (engine->active_viewport->renderer->type) {
        case RENDERER_RT: onResizeRT(engine); break;
        case RENDERER_RC: onResizeRC(engine); break;
    }
    updateHUDDimensions(engine->hud, engine->frame_buffer);
}

void toggleControllerMode(Engine* engine) {
    Viewport* viewport = engine->active_viewport;
    bool in_fps_mode = viewport->controller->type == CONTROLLER_FPS;
    viewport->controller = in_fps_mode ?
            &engine->controllers.orb->controller :
            &engine->controllers.fps->controller;
    setControllerModeInHUD(engine->hud, !in_fps_mode);
}

void updateAndRender(Engine* engine) {
    perfStartFrame(engine->perf);

    HUD* hud = engine->hud;
    Perf* perf = engine->perf;
    Mouse* mouse = engine->mouse;
    Keyboard* keyboard = engine->keyboard;
    Viewport* viewport = engine->active_viewport;
    Renderer* renderer = viewport->renderer;
    Controller* controller = viewport->controller;
    FpsController* fps = engine->controllers.fps;
    OrbController* orb = engine->controllers.orb;
    f32 dt = (f32)perf->delta.seconds;

    if (mouse->wheel.changed) {
        mouse->wheel.changed = false;
        switch (controller->type) {
            case CONTROLLER_FPS: onMouseScrolledFps(fps, mouse); break;
            case CONTROLLER_ORB: onMouseScrolledOrb(orb, mouse); break;
        }
        mouse->wheel.scroll = 0;
    }

    if (mouse->coords.relative.changed) {
        mouse->coords.relative.changed = false;
        switch (controller->type) {
            case CONTROLLER_FPS: onMouseMovedFps(fps, mouse); break;
            case CONTROLLER_ORB: onMouseMovedOrb(orb, mouse); break;
        }
        mouse->coords.relative.x = 0;
        mouse->coords.relative.y = 0;
    }

    switch (controller->type) {
        case CONTROLLER_FPS: onUpdateFps(fps, keyboard, dt); break;
        case CONTROLLER_ORB: onUpdateOrb(orb, keyboard, dt); break;
    }

    if (controller->changed.fov) {
        controller->changed.fov = false;
        switch (renderer->type) {
            case RENDERER_RT: onZoomRT(engine); break;
            case RENDERER_RC: onZoomRC(engine); break;
        }
    }

    if (controller->changed.orientation) {
        controller->changed.orientation = false;
        switch (renderer->type) {
            case RENDERER_RT: onRotateRT(engine); break;
            case RENDERER_RC: onRotateRC(engine); break;
        }
    }

    if (controller->changed.position) {
        controller->changed.position = false;
        switch (renderer->type) {
            case RENDERER_RT: onMoveRT(engine); break;
            case RENDERER_RC: onMoveRC(engine); break;
        }
    }

    switch (renderer->type) {
        case RENDERER_RT: onRenderRT(engine); break;
        case RENDERER_RC: onRenderRC(engine); break;
    }

    perfEndFrame(perf);
    if (hud->is_visible) {
        if (!perf->accum.frames) updateHUDCounters(hud, perf);
        drawText(engine->frame_buffer, hud->text, HUD_COLOR, engine->frame_buffer->width - HUD_RIGHT - HUD_WIDTH, HUD_TOP);
    }

//    if (buttons.first.is_pressed) engine.renderer = &ray_tracer.renderer;
//    if (buttons.second.is_pressed) engine.renderer = &ray_caster.base;


    if (keyboard->hud.is_pressed) {
        keyboard->hud.is_pressed = false;
        hud->is_visible = !hud->is_visible;
    }

    if (mouse->double_clicked) {
        mouse->double_clicked = false;
        toggleControllerMode(engine);
    }
}

Engine* createEngine(
    UpdateWindowTitle updateWindowTitle,
    PrintDebugString printDebugString,
    GetTicks getTicks,
    u64 ticks_per_second
) {
    Engine* engine = Alloc(Engine);
    engine->is_running = true;

    engine->printDebugString = printDebugString;
    engine->updateWindowTitle = updateWindowTitle;

    engine->mouse = createMouse();
    engine->keyboard = createKeyboard();

    engine->hud = createHUD();
    engine->hud->debug_perf = createPerf(getTicks, ticks_per_second);
    engine->perf = createPerf(getTicks, ticks_per_second);

    engine->scene = createScene();
    engine->frame_buffer = createFrameBuffer();

    engine->controllers.fps = createFpsController(engine->scene->camera);
    engine->controllers.orb = createOrbController(engine->scene->camera);

    engine->renderers.ray_tracer = createRayTracer(engine);
    engine->renderers.ray_caster = createRayCaster(engine);

    engine->active_viewport = Alloc(Viewport);
    engine->active_viewport->controller = &engine->controllers.orb->controller;
    engine->active_viewport->renderer = &engine->renderers.ray_tracer->renderer;

    engine->scene->camera->transform->position->x = 5;
    engine->scene->camera->transform->position->y = 5;
    engine->scene->camera->transform->position->z = -10;
    engine->active_viewport->controller->changed.position = true;

    return engine;
}