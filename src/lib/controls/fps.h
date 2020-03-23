#pragma once

#include "lib/core/types.h"
#include "lib/math/math1D.h"
#include "lib/nodes/camera.h"
#include "lib/controls/controls.h"

typedef struct Zoom {
    f32 sensitivity, in;
    bool changed;
} Zoom;
void onZoom(f32 in, Zoom* zoom) {
    zoom->in += in * zoom->sensitivity;
    zoom->changed = true;
}

typedef struct Orientation {
    f32 sensitivity, yaw, pitch;
    bool changed;
} Orientation;
void onOrient(f32 yaw, f32 pitch, Orientation* orientation) {
    orientation->yaw += yaw * orientation->sensitivity;
    orientation->pitch += pitch * orientation->sensitivity;
    orientation->changed = true;
}

typedef struct Velocity {
    f32 x, y, z, maximum;
} Velocity;

typedef struct Acceleration {
    f32 current, maximum;
} Acceleration;

typedef struct FpsController3D {
    Zoom zoom;
    Velocity velocity;
    Orientation orientation;
    Acceleration acceleration;
} FpsController3D;

void initFpsController(FpsController3D* fps_controller) {
    fps_controller->velocity.maximum = 8;
    fps_controller->acceleration.maximum = 35;
    fps_controller->orientation.sensitivity = 7 / 10000.0f;
    fps_controller->zoom.sensitivity = 1;
    fps_controller->zoom.changed = fps_controller->orientation.changed = false;
}

void zoom(Camera3D* camera, FpsController3D* fps_controller) {
    camera->focal_length += fps_controller->zoom.in;

    fps_controller->zoom.in = 0;
    fps_controller->zoom.changed = false;
}

void look(Camera3D* camera, FpsController3D* fps_controller) {
    rotate3D(fps_controller->orientation.yaw,
             fps_controller->orientation.pitch,
             0,
             camera->transform);

    fps_controller->orientation.yaw = 0;
    fps_controller->orientation.pitch = 0;
    fps_controller->orientation.changed = false;
}

void move(Camera3D* camera, FpsController3D* fps_controller, Keyboard* keyboard, Buttons* buttons, f32 delta_time) {
    vec3.x = vec3.y = vec3.z = 0;

    // Compute velocity delta:
    if (keyboard->pressed & buttons->FORWARD) vec3.z += fps_controller->velocity.maximum;
    if (keyboard->pressed & buttons->BACKWARD) vec3.z -= fps_controller->velocity.maximum;
    if (keyboard->pressed & buttons->RIGHT) vec3.x += fps_controller->velocity.maximum;
    if (keyboard->pressed & buttons->LEFT) vec3.x -= fps_controller->velocity.maximum;
    if (keyboard->pressed & buttons->UP) vec3.y += fps_controller->velocity.maximum;
    if (keyboard->pressed & buttons->DOWN) vec3.y -= fps_controller->velocity.maximum;

    // Update current velocity based on deltas of velocity and time:
    fps_controller->acceleration.current = fps_controller->acceleration.maximum * delta_time;
    fps_controller->velocity.x = approach(fps_controller->velocity.x, vec3.x, fps_controller->acceleration.current);
    fps_controller->velocity.y = approach(fps_controller->velocity.y, vec3.y, fps_controller->acceleration.current);
    fps_controller->velocity.z = approach(fps_controller->velocity.z, vec3.z, fps_controller->acceleration.current);

    // Compute movement delta (axis-aligned):
    vec3.x = fps_controller->velocity.x * delta_time;
    vec3.y = fps_controller->velocity.y * delta_time;
    vec3.z = fps_controller->velocity.z * delta_time;

    // Rotate movement delta:
    imul3D(&vec3, camera->transform->yaw);

    // Apply movement delta to the current camera position:
    iadd3D(camera->position, &vec3);
}