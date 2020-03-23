#pragma once
#include "engine/core/types.h"
#include "core.h"
#include "engine/math/math2D.h"


typedef struct Controller {
    u8 rotated, moved;
    f32 rotation_speed, movement_speed;
    f32 rotation_amount, movement_amount;
    Vector2* right;
    Vector2* forward;
} Controller;
Controller controller = {0, 0, 0.001f, 0.005f, 0, 0, &camera.matrix.i, &camera.matrix.j};

void rotateCamera() {

    rotateMatrix2D(&camera.matrix, (mouse.current_position.x - mouse.prior_position.x) * controller.rotation_speed);
}

void on_key_pressed() {
    if (keyboard.pressed & FORWARD) {
        controller.moved = TRUE;
        camera.position->x += controller.forward->x * controller.movement_speed;
        camera.position->y += controller.forward->y * controller.movement_speed;
    }

    if (keyboard.pressed & BACKWARD) {
        controller.moved = TRUE;
        camera.position->x -= controller.forward->x * controller.movement_speed;
        camera.position->y -= controller.forward->y * controller.movement_speed;
    }

    if (keyboard.pressed & RIGHT) {
        controller.moved = TRUE;
        camera.position->x += controller.right->x * controller.movement_speed;
        camera.position->y += controller.right->y * controller.movement_speed;
    }

    if (keyboard.pressed & LEFT) {
        controller.moved = TRUE;
        camera.position->x -= controller.right->x * controller.movement_speed;
        camera.position->y -= controller.right->y * controller.movement_speed;
    }
}

void init_core2D() {
    setMatrix2x2ToIdentity(&camera.matrix);
}