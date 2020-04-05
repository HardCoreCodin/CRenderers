#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/math/math1D.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"
#include "lib/nodes/transform.h"
#include "lib/memory/allocators.h"

typedef struct Zoom {
    f32 sensitivity, in;
    bool changed;
} Zoom;
Zoom zoom = {1 / 10.f, 0, false};

typedef struct Orientation {
    f32 sensitivity, yaw, pitch;
    bool changed;
} Orientation;
Orientation orientation = {7 / 10000.0f, 0, 0, false};

typedef struct Pan   {
    f32 sensitivity, right, up;
    bool changed;
} Pan;
Pan pan = {1 / 100.0f, 0, 0, false};

typedef struct Dolly {
    f32 sensitivity, in, current, factor;
    bool changed;
} Dolly;
Dolly dolly = {1, 0, -5, 5, false};

typedef struct Orbit {
    f32 sensitivity, azimuth, elevation;
    bool changed;
} Orbit;
Orbit orbit = {1 / 1000.0f, 0, 0, false};

void onZoom(f32 in) {
    zoom.in += in * zoom.sensitivity;
    zoom.changed = true;
}

void onOrient(s16 yaw, s16 pitch) {
    orientation.yaw += (f32)yaw * orientation.sensitivity;
    orientation.pitch += (f32)pitch * orientation.sensitivity;
    orientation.changed = true;
}

void updateZoom(f32* focal_length) {
    *focal_length += zoom.in;

    zoom.in = 0;
    zoom.changed = false;
}

void updateOrientation3D(Transform3D* transform) {
    rotate3D(orientation.yaw, orientation.pitch,0, transform);

    orientation.yaw = 0;
    orientation.pitch = 0;
    orientation.changed = false;
}

void updateOrientation2D(Transform2D* transform) {
    rotate2D(orientation.yaw, transform->rotation);
    orientation.yaw = 0;
    orientation.changed = false;
}

#define VELOCITY 8
#define ACCELERATION 20
Vector3 current_velocity, target_velocity, change_in_position;

void onMove3D(Matrix3x3* rotation_matrix, Vector3* position, f32 delta_time) {
    // Compute the target velocity:
    target_velocity.x = target_velocity.y = target_velocity.z = 0;
    if (buttons.right.is_pressed) target_velocity.x += VELOCITY;
    if (buttons.left.is_pressed) target_velocity.x -= VELOCITY;
    if (buttons.up.is_pressed) target_velocity.y += VELOCITY;
    if (buttons.down.is_pressed) target_velocity.y -= VELOCITY;
    if (buttons.forward.is_pressed) target_velocity.z += VELOCITY;
    if (buttons.back.is_pressed) target_velocity.z -= VELOCITY;

    // Update the current velocity:
    f32 change_in_velocity = ACCELERATION * delta_time;
    approach(&current_velocity.x, target_velocity.x, change_in_velocity);
    approach(&current_velocity.y, target_velocity.y, change_in_velocity);
    approach(&current_velocity.z, target_velocity.z, change_in_velocity);

    // Update the current position:
    scale3D(&current_velocity, delta_time, &change_in_position);
    imul3D(&change_in_position, rotation_matrix);
    iadd3D(position, &change_in_position);
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

void onPan(s16 right, s16 up) {
    pan.right += (f32)right * pan.sensitivity;
    pan.up += (f32)up * pan.sensitivity;
    pan.changed = true;
}

void onDolly(f32 in) {
    dolly.in += in * dolly.sensitivity;
    dolly.changed = true;
}

void onOrbit(s16 azimuth, s16 elevation) {
    orbit.azimuth += (f32)azimuth * orbit.sensitivity;
    orbit.elevation += (f32)elevation * orbit.sensitivity;
    orbit.changed = true;
}

Vector3 target_position, target_direction;

void updateDolly(Transform3D* transform) {
    scale3D(transform->forward, dolly.factor, &target_direction);
    add3D(transform->position, &target_direction, &target_position);

    dolly.current += dolly.in;
    if (dolly.current > 0)
        dolly.factor = 1 / dolly.current;
    else
        dolly.factor = -dolly.current;

    scale3D(transform->forward, dolly.factor, &target_direction);
    sub3D(&target_position, &target_direction, transform->position);

    dolly.in = 0;
    dolly.changed = false;
}

void updatePan3D(Transform3D* transform) {
    scale3D(transform->right, pan.right, &vec3);
    iadd3D(transform->position, &vec3);

    scale3D(transform->up, pan.up, &vec3);
    iadd3D(transform->position, &vec3);

    pan.right = pan.up = 0;
    pan.changed = false;
}

void updateOrbit3D(Transform3D* transform) {
    scale3D(transform->forward, dolly.factor, &target_direction);
    add3D(transform->position, &target_direction, &target_position);

    rotate3D(orbit.azimuth, orbit.elevation,0,transform);

    scale3D(transform->forward, dolly.factor, &target_direction);
    sub3D(&target_position, &target_direction, transform->position);

    orbit.azimuth = orbit.elevation = 0;
    orbit.changed = false;
}

void updatePan2D(Transform2D* transform) {
    scale2D(transform->right, pan.right, &vec2);
    iadd2D(transform->position, &vec2);
    pan.right = 0;
    pan.changed = false;
}

void updateOrbit2D(Transform2D* transform) {
    rotate2D(orbit.azimuth, transform->rotation);
    orbit.azimuth = 0;
    orbit.changed = false;
}