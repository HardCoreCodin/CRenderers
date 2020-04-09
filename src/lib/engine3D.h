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

#ifdef RAY_TRACER
#include "lib/render/raytracing/raytracer.h"
#elif defined(RAY_CASTER)
#include "lib/render/raycasting/raycaster.h"
#endif

#define VELOCITY 8
#define ACCELERATION 20
Vector3 current_velocity, target_velocity, change_in_position;
Vector3 target_position, target_direction;

typedef struct Sensitivity {
    f32 zoom, dolly, orbit, orient, pan;
} Sensitivity;

typedef struct RenderEngine {
    bool is_running, in_fps_mode;
    Sensitivity sensitivity;
} RenderEngine;
RenderEngine engine;

void initRenderEngine() {
    initPerf(&perf);

    initHUD();
    initMouse();
    initButtons();

    initScene();
    initFrameBuffer();
    initRenderer();

    engine.is_running = true;
    engine.in_fps_mode = false;

    engine.sensitivity.dolly = 1;
    engine.sensitivity.zoom = 1 / 10.0f;
    engine.sensitivity.pan = 1 / 100.0f;
    engine.sensitivity.orbit = 1 / 1000.0f;
    engine.sensitivity.orient = 7 / 10000.0f;
}

void OnFrameBufferResized() {
    updateHUDDimensions();
    onResized();
}

void update() {
    bool moved = false;

    Transform3D* transform = renderer.camera.transform;
    Vector3* up = transform->up;
    Vector3* right = transform->right;
    Vector3* forward = transform->forward;
    Vector3* position = transform->position;

    if (mouse.wheel.was_scrolled) {
        mouse.wheel.was_scrolled = false;

        if (engine.in_fps_mode) {
            renderer.camera.focal_length += mouse.wheel.scroll_amount * engine.sensitivity.zoom;
            onZoomed();
        } else {
            scale3D(forward, controller.dolly.factor, &target_direction);
            add3D(position, &target_direction, &target_position);

            f32 dolly = mouse.wheel.scroll_amount * engine.sensitivity.dolly;
            if (dolly > 0)
                controller.dolly.factor = 1 / dolly;
            else
                controller.dolly.factor = -dolly;

            scale3D(forward, controller.dolly.factor, &target_direction);
            sub3D(&target_position, &target_direction, position);

            moved = true;
        }

        mouse.wheel.scroll_amount = 0;
    }

    if (mouse.has_moved) {
        mouse.has_moved = false;

        f32 d_left = -mouse.coords.relative.x;
        f32 d_down = -mouse.coords.relative.y;

        if (engine.in_fps_mode) { // Look:
            f32 yaw = d_left * engine.sensitivity.orient;
            f32 pitch = d_down * engine.sensitivity.orient;
            rotate3D(yaw, pitch,0, transform);

            moved = true;
        } else if (mouse.buttons.right.is_down) { // Orbit:
            scale3D(forward, controller.dolly.factor, &target_direction);
            add3D(position, &target_direction, &target_position);

            f32 yaw = d_left * engine.sensitivity.orbit;
            f32 pitch = d_down * engine.sensitivity.orbit;
            rotate3D(yaw, pitch,0, transform);

            scale3D(forward, controller.dolly.factor, &target_direction);
            sub3D(&target_position, &target_direction, position);

            moved = true;
        } else if (mouse.buttons.middle.is_down) { // Pan:
            scale3D(right, d_left * engine.sensitivity.pan, &vec3);
            iadd3D(position, &vec3);

            scale3D(up, -d_down * engine.sensitivity.pan, &vec3);
            iadd3D(position, &vec3);

            moved = true;
        }

        mouse.coords.relative.x = 0;
        mouse.coords.relative.y = 0;
    }

    if (buttons.hud.is_pressed) {
        buttons.hud.is_pressed = false;
        hud.is_visible = !hud.is_visible;
    }



    // Compute the target velocity:
    target_velocity.x = target_velocity.y = target_velocity.z = 0;
    if (buttons.right.is_pressed) target_velocity.x += VELOCITY;
    if (buttons.left.is_pressed) target_velocity.x -= VELOCITY;
    if (buttons.up.is_pressed) target_velocity.y += VELOCITY;
    if (buttons.down.is_pressed) target_velocity.y -= VELOCITY;
    if (buttons.forward.is_pressed) target_velocity.z += VELOCITY;
    if (buttons.back.is_pressed) target_velocity.z -= VELOCITY;

    // Update the current velocity:
    f32 delta_time = (f32) perf.delta.seconds;
    if (delta_time > 1)
        delta_time = 1;
    f32 change_in_velocity = ACCELERATION * delta_time;
    approach(&current_velocity.x, target_velocity.x, change_in_velocity);
    approach(&current_velocity.y, target_velocity.y, change_in_velocity);
    approach(&current_velocity.z, target_velocity.z, change_in_velocity);

    // Update the current position:
    if (current_velocity.x ||
        current_velocity.y ||
        current_velocity.z) {
        scale3D(&current_velocity, delta_time, &change_in_position);
        imul3D(&change_in_position, renderer.camera.transform->yaw);
        iadd3D(position, &change_in_position);

        moved = true;
    }

    if (moved)
        onMoved();

    if (mouse.double_click.was_actioned) {
        mouse.double_click.was_actioned = false;
        engine.in_fps_mode = !engine.in_fps_mode;
        setControllerModeInHUD(engine.in_fps_mode);
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