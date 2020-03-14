#pragma once
#include "types.h"
#include "core.h"
#include "math3D.h"

typedef struct Camera {
    f32 focal_length;
    Matrix3x3 matrix;
    Vector3* position;
} Camera;

Camera camera = {1};

typedef struct Rotation {u8 changed; f32 speed, yaw, pitch;} Rotation;
typedef struct Movement {u8 changed; f32 speed, x, y, z;} Movement;
typedef struct Zoom     {u8 changed; f32 speed, in;} Zoom;

typedef struct Controller {
    Rotation rotation;
    Movement movement;
    Zoom zoom;
} Controller;
Controller controller;

void init_core3D() {
    controller.movement.speed = 10;
    controller.rotation.speed = 0.05f;
    controller.zoom.speed = 1;
    controller.zoom.changed = controller.movement.changed = controller.rotation.changed = FALSE;
    setMatrix3x3ToIdentity(&camera.matrix);
}

void onMousePositionChanged(f32 dx, f32 dy) {
    controller.rotation.yaw -= dx * controller.rotation.speed;
    controller.rotation.pitch -= dy * controller.rotation.speed;
    controller.rotation.changed = TRUE;
}
void onMousePositionChangeHandled() {
    controller.rotation.yaw = 0;
    controller.rotation.pitch = 0;
    controller.rotation.changed = FALSE;
}

void onMouseWheelChanged(f32 amount) {
    controller.zoom.in += amount * controller.zoom.speed;
    controller.zoom.changed = TRUE;
}
void onMouseWheelChangeHandled() {
    controller.zoom.in = 0;
    controller.zoom.changed = FALSE;
}

void processKeyboardInputs(f32 delta_time) {
    controller.movement.changed = FALSE;
    controller.movement.x = controller.movement.y = controller.movement.z = 0;

    if (keyboard.pressed & FORWARD) {
        controller.movement.changed = TRUE;
        controller.movement.x += controller.movement.speed * yaw_matrix.k.x;
        controller.movement.z += controller.movement.speed * yaw_matrix.k.z;
    }
    if (keyboard.pressed & BACKWARD) {
        controller.movement.changed = TRUE;
        controller.movement.x -= controller.movement.speed * yaw_matrix.k.x;
        controller.movement.z -= controller.movement.speed * yaw_matrix.k.z;
    }
    if (keyboard.pressed & RIGHT) {
        controller.movement.changed = TRUE;
        controller.movement.x += controller.movement.speed * yaw_matrix.k.z;
        controller.movement.z -= controller.movement.speed * yaw_matrix.k.x;
    }
    if (keyboard.pressed & LEFT) {
        controller.movement.changed = TRUE;
        controller.movement.x -= controller.movement.speed * yaw_matrix.k.z;
        controller.movement.z += controller.movement.speed * yaw_matrix.k.x;
    }
    if (keyboard.pressed & UP) {
        controller.movement.changed = TRUE;
        controller.movement.y += controller.movement.speed;
    }
    if (keyboard.pressed & DOWN) {
        controller.movement.changed = TRUE;
        controller.movement.y -= controller.movement.speed;
    }

    if (controller.movement.changed) {
        camera.position->x += controller.movement.x * delta_time;
        camera.position->y += controller.movement.y * delta_time;
        camera.position->z += controller.movement.z * delta_time;
    }
}