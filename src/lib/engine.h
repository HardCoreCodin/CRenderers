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
#include "lib/display/viewport.h"
#include "lib/engine.h"
#include "lib/render/raytracing/raytracer.h"
#include "lib/render/raycasting/raycaster.h"

static Engine engine;

char* getTitle() {
    return viewport.renderer->title;
}

void onUpdateAndRender() {
    PERF_START_FRAME

    updateController(viewport.controller);
    viewport.refresh();

    if (fps.controller.changed.fov) viewport.renderer->zoom(&fps.controller);
    if (fps.controller.changed.orientation) viewport.renderer->rotate(&fps.controller);
    if (fps.controller.changed.position) viewport.renderer->move(&fps.controller);

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
        viewport.toggleControllerMode();
    }
}

void initEngine(Callback updateWindowTitle) {
    initPerf(&perf);

    initHUD();
    initMouse();
    initButtons();

    initScene(&engine.scene);
    initFpsController(&engine.scene.camera);
    initOrbController(&engine.scene.camera);

    initFrameBuffer();
    initRayTracer(&engine);
    initRayCaster(&engine);
    initViewport(&ray_tracer.renderer, &orb.controller);

    engine.getTitle = getTitle;
    engine.updateAndRender = onUpdateAndRender;
    engine.is_running = true;

    viewport.updateWindowTitle = updateWindowTitle;

    engine.scene.camera.transform->position->x = 5;
    engine.scene.camera.transform->position->y = 5;
    engine.scene.camera.transform->position->z = -10;
    orb.controller.changed.position = true;
    OrbOnReset();
}