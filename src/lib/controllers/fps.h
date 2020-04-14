#pragma once

#include "lib/core/types.h"
#include "lib/math/math1D.h"
#include "lib/math/math3D.h"
#include "lib/controllers/base.h"

typedef struct {
    f32 max_velocity, max_acceleration, orientation_speed, zoom_speed;
    f32 vx, vy, vz, tx, ty, tz, dx, dy, dz, dv, dt;
    Controller controller;
} FpsController;
FpsController fps = {
        .max_velocity = 8,
        .max_acceleration = 20,
        .orientation_speed = 7.0f / 10000,
        .zoom_speed = 1,
        .vx = 0,
        .vy = 0,
        .vz = 0
};

void FpsOnMouseScrolled() {
    fps.controller.camera->focal_length += fps.zoom_speed * mouse.wheel.scroll;
    fps.controller.changed.fov = true;
}

void FpsOnMouseMoved() {
    f32 yaw   = fps.orientation_speed * (f32)-mouse.coords.relative.x;
    f32 pitch = fps.orientation_speed * (f32)-mouse.coords.relative.y;

    if (yaw) yaw3D(yaw, fps.controller.camera->transform->yaw);
    if (pitch) pitch3D(pitch, fps.controller.camera->transform->pitch);
    matMul3D(fps.controller.camera->transform->pitch,
             fps.controller.camera->transform->yaw,
             fps.controller.camera->transform->rotation);

    fps.controller.changed.orientation = true;
}

void FpsOnUpdate() {
    // Compute the target velocity:
    fps.tx = fps.ty = fps.tz = 0;
    if (buttons.right.is_pressed) fps.tx += fps.max_velocity;
    if (buttons.left.is_pressed) fps.tx -= fps.max_velocity;
    if (buttons.up.is_pressed) fps.ty += fps.max_velocity;
    if (buttons.down.is_pressed) fps.ty -= fps.max_velocity;
    if (buttons.forward.is_pressed) fps.tz += fps.max_velocity;
    if (buttons.back.is_pressed) fps.tz -= fps.max_velocity;

    // Update the current velocity:
    fps.dt = perf.delta.seconds > 1 ? 1 : (f32)perf.delta.seconds;
    fps.dv = fps.dt * fps.max_acceleration;
    approach(&fps.vx, fps.tx, fps.dv);
    approach(&fps.vy, fps.ty, fps.dv);
    approach(&fps.vz, fps.tz, fps.dv);

    fps.controller.changed.position = fps.vx || fps.vy || fps.vz;
    if (fps.controller.changed.position) {
        // Update the current position:
        fps.dx = fps.vx * fps.dt;
        fps.dy = fps.vy * fps.dt;
        fps.dz = fps.vz * fps.dt;

        fps.controller.camera->transform->position->y += fps.dy;
        fps.controller.camera->transform->position->x +=
        fps.controller.camera->transform->yaw.x_axis->x * fps.dx +
        fps.controller.camera->transform->yaw.z_axis->x * fps.dz;
        fps.controller.camera->transform->position->z +=
        fps.controller.camera->transform->yaw.x_axis->z * fps.dx +
        fps.controller.camera->transform->yaw.z_axis->z * fps.dz;
    }
}