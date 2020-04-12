#pragma once

#include "lib/core/types.h"
#include "lib/math/math1D.h"
#include "lib/math/math3D.h"
#include "lib/controllers/base.h"

typedef struct { f32 velocity, acceleration, orientation, zoom; } FpsControllerSpeed;
typedef struct {
    FpsControllerSpeed speed;
    Controller controller;
    Matrix3x3 yaw, pitch, rotation;
    Vector3 *current_velocity, *target_velocity, *movement, *up, *right, *forward, *pan, *position;
} FpsController;

FpsController fps = {8, 20, 7.0f / 10000, 1};

void FpsOnMouseScrolled() {
    fps.controller.camera->focal_length += fps.speed.zoom * mouse.wheel.scroll;
    fps.controller.changed.fov = true;
}

void FpsOnMouseMoved() {
    f32 yaw   = fps.speed.orientation * (f32)-mouse.coords.relative.x;
    f32 pitch = fps.speed.orientation * (f32)-mouse.coords.relative.y;

    if (yaw) yaw3D(yaw, fps.yaw);
    if (pitch) pitch3D(pitch, fps.pitch);
    matMul3D(fps.pitch, fps.yaw, fps.rotation);

    fps.controller.changed.orientation = true;
}

void FpsOnUpdate() {
    // Compute the target velocity:
    fps.target_velocity->x = fps.target_velocity->y = fps.target_velocity->z = 0;
    if (buttons.right.is_pressed) fps.target_velocity->x += fps.speed.velocity;
    if (buttons.left.is_pressed) fps.target_velocity->x -= fps.speed.velocity;
    if (buttons.up.is_pressed) fps.target_velocity->y += fps.speed.velocity;
    if (buttons.down.is_pressed) fps.target_velocity->y -= fps.speed.velocity;
    if (buttons.forward.is_pressed) fps.target_velocity->z += fps.speed.velocity;
    if (buttons.back.is_pressed) fps.target_velocity->z -= fps.speed.velocity;

    // Update the current velocity:
    f32 delta_time = (f32) perf.delta.seconds;
    if (delta_time > 1)
        delta_time = 1;

    f32 change_in_velocity = fps.speed.acceleration * delta_time;
    approach(&fps.current_velocity->x, fps.target_velocity->x, change_in_velocity);
    approach(&fps.current_velocity->y, fps.target_velocity->y, change_in_velocity);
    approach(&fps.current_velocity->z, fps.target_velocity->z, change_in_velocity);

    if (fps.current_velocity->x ||
        fps.current_velocity->y ||
        fps.current_velocity->z) {

        // Update the current position:
        scale3D(fps.current_velocity, delta_time, fps.movement);
        imul3D(fps.movement, fps.yaw);
        iadd3D(fps.position, fps.movement);

        fps.controller.changed.position = true;
    }
}

void FpsOnReset() {
    fps.current_velocity->x = 0;
    fps.current_velocity->y = 0;
    fps.current_velocity->z = 0;

    fps.yaw = fps.controller.camera->transform->yaw;
    fps.pitch = fps.controller.camera->transform->pitch;
    fps.rotation = fps.controller.camera->transform->rotation;

    fps.up = fps.controller.camera->transform->up;
    fps.right = fps.controller.camera->transform->right;
    fps.forward = fps.controller.camera->transform->forward;
    fps.position = fps.controller.camera->transform->position;
}

void initFpsController(Camera* camera) {
    fps.movement = Alloc(Vector3);
    fps.target_velocity = Alloc(Vector3);
    fps.current_velocity = Alloc(Vector3);

    initController(&fps.controller, camera, FpsOnReset, FpsOnUpdate, FpsOnMouseMoved, FpsOnMouseScrolled);
}
