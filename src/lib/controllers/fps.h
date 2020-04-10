#pragma once

#include "lib/core/types.h"
#include "lib/math/math1D.h"
#include "lib/math/math3D.h"
#include "lib/controllers/base.h"

#define FPS_CONTROLLER_INITIAL_VELOCITY 8.0f
#define FPS_CONTROLLER_INITIAL_ACCELERATION 20.0f
#define FPS_CONTROLLER_INITIAL_MOUSE_MOVEMENT_SPEED 0.0007f
#define FPS_CONTROLLER_INITIAL_MOUSE_SCROLL_SPEED 0.1f
#define FPS_CONTROLLER_INITIAL_MOUSE_MOVEMENT_SPEED 0.0007f

typedef struct {
    Controller controller;
    Camera3D *camera;

    Vector3 *up, *right, *forward, *position, *current_velocity, *target_velocity;
    Matrix3x3 *yaw, *pitch, *rotation;

    f32 velocity, acceleration;

    Callback move, rotate, zoom;
} FpsController;

static FpsController fps;

void FpsOnMouseScrolled() {
    fps.camera->focal_length += fps.controller.mouse_scroll_speed * mouse.wheel.scroll_amount;
    fps.controller.zoomed = true;
}

void FpsOnMouseMoved() {
    f32 yaw   = -fps.controller.mouse_movement_speed * (f32)mouse.coords.relative.x;
    f32 pitch = -fps.controller.mouse_movement_speed * (f32)mouse.coords.relative.y;

    if (yaw) yaw3D(yaw, fps.yaw);
    if (pitch) pitch3D(pitch, fps.pitch);
    matMul3D(fps.pitch, fps.yaw, fps.rotation);

    fps.controller.rotated = true;
}

void FpsOnUpdate() {
    // Compute the target velocity:
    fps.target_velocity->x = fps.target_velocity->y = fps.target_velocity->z = 0;
    if (buttons.right.is_pressed) fps.target_velocity->x += fps.velocity;
    if (buttons.left.is_pressed) fps.target_velocity->x -= fps.velocity;
    if (buttons.up.is_pressed) fps.target_velocity->y += fps.velocity;
    if (buttons.down.is_pressed) fps.target_velocity->y -= fps.velocity;
    if (buttons.forward.is_pressed) fps.target_velocity->z += fps.velocity;
    if (buttons.back.is_pressed) fps.target_velocity->z -= fps.velocity;

    // Update the current velocity:
    f32 delta_time = (f32) perf.delta.seconds;
    if (delta_time > 1)
        delta_time = 1;

    f32 change_in_velocity = fps.acceleration * delta_time;
    approach(&fps.current_velocity->x, fps.target_velocity->x, change_in_velocity);
    approach(&fps.current_velocity->y, fps.target_velocity->y, change_in_velocity);
    approach(&fps.current_velocity->z, fps.target_velocity->z, change_in_velocity);

    if (fps.current_velocity->x ||
        fps.current_velocity->y ||
        fps.current_velocity->z) {
        fps.controller.moved = true;

        // Update the current position:
        scale3D(fps.current_velocity, delta_time, fps.controller.movement);
        imul3D(fps.controller.movement, fps.yaw);
        iadd3D(fps.position, fps.controller.movement);
    }

    if (fps.controller.zoomed) fps.zoom(&fps.controller);
    if (fps.controller.rotated) fps.rotate(&fps.controller);
    if (fps.controller.moved) fps.move(&fps.controller);
}

void initFpsController(Camera3D* camera, Callback zoom, Callback move, Callback rotate) {
    fps.camera = camera;

    fps.zoom = zoom;
    fps.move = move;
    fps.rotate = rotate;

    fps.yaw = camera->transform->yaw;
    fps.pitch = camera->transform->pitch;
    fps.position = camera->transform->position;
    fps.rotation = camera->transform->rotation;
    fps.up = camera->transform->up;
    fps.right = camera->transform->right;
    fps.forward = camera->transform->forward;

    fps.velocity = FPS_CONTROLLER_INITIAL_VELOCITY;
    fps.acceleration = FPS_CONTROLLER_INITIAL_ACCELERATION;
    fps.target_velocity = (Vector3*)allocate(sizeof(Vector3));
    fps.current_velocity = (Vector3*)allocate(sizeof(Vector3));
    fps.current_velocity->x = fps.current_velocity->y = fps.current_velocity->z = 0;

    initController(
            &fps.controller,
            FpsOnUpdate,
            FpsOnMouseMoved,
            FpsOnMouseScrolled,
            FPS_CONTROLLER_INITIAL_MOUSE_MOVEMENT_SPEED,
            FPS_CONTROLLER_INITIAL_MOUSE_SCROLL_SPEED);
}
