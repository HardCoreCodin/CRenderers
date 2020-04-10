#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/memory/allocators.h"
#include "lib/controllers/base.h"

#define ORB_CONTROLLER_INITIAL_MOUSE_PAN_SPEED 0.01f
#define ORB_CONTROLLER_INITIAL_MOUSE_SCROLL_SPEED 1.0f
#define ORB_CONTROLLER_INITIAL_MOUSE_MOVEMENT_SPEED 0.001f

typedef struct { f32 factor, amount; } OrbControllerDolly;
typedef struct {
    Controller controller;
    Camera3D *camera;

    Matrix3x3 *yaw, *pitch, *rotation;
    Vector3 *up, *right, *forward, *position, *target_position;
    OrbControllerDolly dolly;

    bool moved, rotated;
    Callback move, rotate, zoom;
} OrbController;

static OrbController orb;

void OrbOnMouseWheelScrolled() {
    orb.dolly.amount += mouse.wheel.scroll_amount * orb.controller.mouse_scroll_speed;
    if (orb.dolly.amount > 0)
        orb.dolly.factor = 1 / orb.dolly.amount;
    else
        orb.dolly.factor = -orb.dolly.amount;

    scale3D(orb.forward, orb.dolly.factor, orb.controller.movement);
    sub3D(orb.target_position, orb.controller.movement, orb.position);

    orb.controller.moved = true;
}

void OrbOnMouseMoved() {
    if (mouse.buttons.right.is_down) { // Orbit:
        f32 yaw   = -orb.controller.mouse_movement_speed * (f32)mouse.coords.relative.x;
        f32 pitch = -orb.controller.mouse_movement_speed * (f32)mouse.coords.relative.y;
        if (yaw) yaw3D(yaw, orb.yaw);
        if (pitch) pitch3D(pitch, orb.pitch);
        matMul3D(orb.pitch, orb.yaw, orb.rotation);

        scale3D(orb.forward, orb.dolly.factor, orb.controller.movement);
        sub3D(orb.target_position, orb.controller.movement, orb.position);

        orb.controller.rotated = true;
    } else if (mouse.buttons.middle.is_down) { // Pan:
        f32 right = -ORB_CONTROLLER_INITIAL_MOUSE_PAN_SPEED * (f32)mouse.coords.relative.x;
        f32 up    =  ORB_CONTROLLER_INITIAL_MOUSE_PAN_SPEED * (f32)mouse.coords.relative.y;

        scale3D(orb.right, right, orb.controller.movement);
        iadd3D(orb.position, orb.controller.movement);
        iadd3D(orb.target_position, orb.controller.movement);

        scale3D(orb.up, up, orb.controller.movement);
        iadd3D(orb.position, orb.controller.movement);
        iadd3D(orb.target_position, orb.controller.movement);

        orb.controller.moved = true;
    }
}

void OrbOnUpdate() {
    if (orb.controller.zoomed) orb.zoom(&orb.controller);
    if (orb.controller.rotated) orb.rotate(&orb.controller);
    if (orb.controller.moved) orb.move(&orb.controller);
}

void initOrbController(Camera3D* camera, Callback zoom, Callback move, Callback rotate) {
    orb.camera = camera;

    orb.zoom = zoom;
    orb.move = move;
    orb.rotate = rotate;

    orb.yaw = camera->transform->yaw;
    orb.pitch = camera->transform->pitch;
    orb.rotation = camera->transform->rotation;
    orb.position = camera->transform->position;
    orb.up = camera->transform->up;
    orb.right = camera->transform->right;
    orb.forward = camera->transform->forward;

    orb.dolly.amount = 0;
    orb.dolly.factor = 1;

    initController(
            &orb.controller,
            OrbOnUpdate,
            OrbOnMouseMoved,
            OrbOnMouseWheelScrolled,
            ORB_CONTROLLER_INITIAL_MOUSE_MOVEMENT_SPEED,
            ORB_CONTROLLER_INITIAL_MOUSE_SCROLL_SPEED);

    orb.target_position = (Vector3*)allocate(sizeof(Vector3));
    orb.controller.movement->z = 1;
    add3D(orb.position, orb.controller.movement, orb.target_position);
}

