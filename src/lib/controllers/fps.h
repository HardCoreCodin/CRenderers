#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

void onMouseScrolledFps(Engine* engine) {
    FpsController* fps = engine->controllers.fps;
    fps->controller.camera->focal_length += fps->zoom_speed * engine->mouse->wheel.scroll;
    fps->controller.changed.fov = true;
}

void onMouseMovedFps(Engine* engine) {
    FpsController* fps = engine->controllers.fps;
    Transform3D* tr = fps->controller.camera->transform;
    f32 speed = fps->orientation_speed;
    f32 x = (f32)engine->mouse->coords.relative.x;
    f32 y = (f32)engine->mouse->coords.relative.y;

    f32 yaw   = speed * -x;
    f32 pitch = speed * -y;

    if (yaw) yaw3D(yaw, tr->yaw);
    if (pitch) pitch3D(pitch, tr->pitch);
    matMul3D(tr->pitch, tr->yaw, tr->rotation);

    fps->controller.changed.orientation = true;
}

void onUpdateFps(Engine* engine) {
    FpsController* fps = engine->controllers.fps;
    Transform3D* tr = fps->controller.camera->transform;
    Keyboard* keyboard = engine->keyboard;

    bool* moved = &fps->controller.changed.position;
    f32 dt = (f32)engine->perf->delta.seconds;
    Vector3* trg_v = fps->target_velocity;
    Vector3* cur_v = fps->current_velocity;
    f32 max_v = fps->max_velocity;

    // Compute the target velocity:
    trg_v->x = trg_v->y = trg_v->z = 0;
    if (keyboard->right.is_pressed  ) trg_v->x += max_v;
    if (keyboard->left.is_pressed   ) trg_v->x -= max_v;
    if (keyboard->up.is_pressed     ) trg_v->y += max_v;
    if (keyboard->down.is_pressed   ) trg_v->y -= max_v;
    if (keyboard->forward.is_pressed) trg_v->z += max_v;
    if (keyboard->back.is_pressed   ) trg_v->z -= max_v;

    // Update the current velocity:
    fps->delta_time = dt > 1 ? 1 : dt;
    f32 dv = fps->delta_time * fps->max_acceleration;
    approach3D(cur_v, trg_v, dv);

    *moved = cur_v->x || cur_v->y || cur_v->z;
    if (*moved) {
        // Update the current position:
        scale3D(cur_v, fps->delta_time, fps->movement);

        Vector3* pos = tr->position;
        Vector3* X = tr->yaw->x_axis;
        Vector3* Z = tr->yaw->z_axis;
        Vector3* M = fps->movement;

        pos->y += M->y;
        pos->x += M->x * X->x + M->z * Z->x;
        pos->z += M->x * X->z + M->z * Z->z;
    }
}

FpsController* createFpsController(Camera* camera) {
    FpsController* fps_controller = Alloc(FpsController);

    fps_controller->controller.on.mouseScrolled = onMouseScrolledFps;
    fps_controller->controller.on.mouseMoved = onMouseMovedFps;
    fps_controller->controller.on.update = onUpdateFps;
    fps_controller->controller.type = CONTROLLER_FPS;
    fps_controller->controller.camera = camera;

    fps_controller->target_velocity = Alloc(Vector3);
    fps_controller->current_velocity = Alloc(Vector3);
    fps_controller->movement = Alloc(Vector3);

    fps_controller->max_velocity = 8;
    fps_controller->max_acceleration = 20;
    fps_controller->orientation_speed = 7.0f / 10000;
    fps_controller->zoom_speed = 1;

    return fps_controller;
}