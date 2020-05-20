#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

typedef struct {
    Controller controller;
    f32 max_velocity, max_acceleration, orientation_speed, zoom_speed, delta_time;
    Vector3 current_velocity, target_velocity, movement;
} FpsController;
FpsController fps = {
        .controller = {.type = CONTROLLER_FPS, .changed = {false, false, false}},
        .max_velocity = 8,
        .max_acceleration = 20,
        .orientation_speed = 7.0f / 10000,
        .zoom_speed = 1,
        .delta_time = 0,
        .current_velocity = {0, 0, 0}
};

void onMouseScrolledFps() {
    fps.controller.camera->focal_length += fps.zoom_speed * mouse.wheel.scroll;
    fps.controller.changed.fov = true;
}

void onMouseMovedFps() {
    f32 yaw   = fps.orientation_speed * (f32)-mouse.coords.relative.x;
    f32 pitch = fps.orientation_speed * (f32)-mouse.coords.relative.y;

    Transform3D* tr = fps.controller.camera->transform;
    if (yaw) yaw3D(yaw, tr->yaw);
    if (pitch) pitch3D(pitch, tr->pitch);
    matMul3D(tr->pitch, tr->yaw, tr->rotation);

    fps.controller.changed.orientation = true;
}

void onUpdateFps() {
    // Compute the target velocity:
    fps.target_velocity.x = fps.target_velocity.y = fps.target_velocity.z = 0;
    if (buttons.right.is_pressed)   fps.target_velocity.x += fps.max_velocity;
    if (buttons.left.is_pressed)    fps.target_velocity.x -= fps.max_velocity;
    if (buttons.up.is_pressed)      fps.target_velocity.y += fps.max_velocity;
    if (buttons.down.is_pressed)    fps.target_velocity.y -= fps.max_velocity;
    if (buttons.forward.is_pressed) fps.target_velocity.z += fps.max_velocity;
    if (buttons.back.is_pressed)    fps.target_velocity.z -= fps.max_velocity;

    Transform3D* tr = fps.controller.camera->transform;

    // Update the current velocity:
    fps.delta_time = perf.delta.seconds > 1 ? 1 : (f32)perf.delta.seconds;
    approach3D(&fps.current_velocity, &fps.target_velocity, fps.delta_time * fps.max_acceleration);
    fps.controller.changed.position = fps.current_velocity.x || fps.current_velocity.y || fps.current_velocity.z;
    if (fps.controller.changed.position) {
        // Update the current position:
        scale3D(&fps.current_velocity, fps.delta_time, &fps.movement);

        Vector3* pos = tr->position;
        Vector3* X = tr->yaw->x_axis;
        Vector3* Z = tr->yaw->z_axis;
        f32 x = fps.movement.x;
        f32 y = fps.movement.y;
        f32 z = fps.movement.z;

        pos->y += y;
        pos->x += X->x*x + Z->x*z;
        pos->z += X->z*x + Z->z*z;
    }
}