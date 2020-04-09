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



typedef struct RenderEngine {
    bool is_running, in_fps_mode;
} RenderEngine;
RenderEngine engine;

void initRenderEngine() {
    initPerf(&perf);

    initHUD();
    initMouse();
    initButtons();

    initFrameBuffer();
    initRenderer();
    initScene();

    engine.is_running = true;
    engine.in_fps_mode = false;
}



Vector2 change_in_position_2D;

void onMove2D(Matrix2x2* rotation_matrix, Vector2* position, f32 delta_time) {
    // Compute the target velocity:
    target_velocity.x = target_velocity.y = 0;
    if (buttons.right.is_pressed) target_velocity.x += VELOCITY;
    if (buttons.left.is_pressed) target_velocity.x -= VELOCITY;
    if (buttons.forward.is_pressed) target_velocity.y += VELOCITY;
    if (buttons.back.is_pressed) target_velocity.y -= VELOCITY;

    // Update the current velocity:
    f32 change_in_velocity = ACCELERATION * delta_time;
    approach(&current_velocity.x, target_velocity.x, change_in_velocity);
    approach(&current_velocity.y, target_velocity.y, change_in_velocity);

    // Update the current position:
    change_in_position_2D.x = current_velocity.x * delta_time;
    change_in_position_2D.y = current_velocity.y * delta_time;
    imul2D(&change_in_position_2D, rotation_matrix);
    iadd2D(position, &change_in_position_2D);
}

void updatePan2D(Transform2D* transform) {
    scale2D(transform->right, controller.pan.right, &vec2);
    iadd2D(transform->position, &vec2);
    controller.pan.right = 0;
    controller.pan.changed = false;
}

void updateOrbit2D(Transform2D* transform) {
    rotate2D(controller.orbit.azimuth, transform->rotation);
    controller.orbit.azimuth = 0;
    controller.orbit.changed = false;
}

void OnFrameBufferResized() {
    updateHUDDimensions();
    onSizeChanged();
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

    f32 delta_time = (f32) perf.delta.seconds;
    if (delta_time > 1)
        delta_time = 1;

    if (engine.in_fps_mode) {
        if (controller.zoom.changed) onZoomChanged();
        if (controller.orientation.changed) onOrientationChanged();
    } else {
        if (controller.orbit.changed) {
            switch (engine.renderer) {
                case RAY_TRACE:
                    updateOrbit3D(renderer.camera.transform);
                    onOrbitRT();
                    break;
                case RAY_CAST:
                    updateOrbit2D(ray_caster.camera.transform);
                    onOrbitRC();
                    break;
            }
        }
        if (controller.pan.changed)
            switch (engine.renderer) {
                case RAY_TRACE:
                    updatePan3D(renderer.camera.transform);
                    break;
                case RAY_CAST:
                    updatePan2D(ray_caster.camera.transform);
                    break;
            }

        if (controller.dolly.changed)
            switch (engine.renderer) {
                case RAY_TRACE:
                    updateDolly(renderer.camera.transform);
                    break;
                case RAY_CAST:
                    break;
            }
    }

    switch (engine.renderer) {
        case RAY_TRACE:

            break;
        case RAY_CAST:
            onMove2D(ray_caster.camera.transform->rotation, ray_caster.camera.transform->position, delta_time);
            break;
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