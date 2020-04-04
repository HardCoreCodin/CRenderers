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

typedef struct Velocity {
    f32 x, y, z, maximum;
} Velocity;
Velocity velocity = {0, 0, 0, 8};

typedef struct Acceleration {
    f32 current, maximum;
} Acceleration;
Acceleration acceleration = {0, 35};

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

void onMove3D(Transform3D* transform, f32 delta_time) {
    vec3.x = vec3.y = vec3.z = 0;

    // Compute velocity delta:
    if (keyboard.keys_pressed & keyboard.keys.FORWARD) vec3.z += velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.BACKWARD) vec3.z -= velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.RIGHT) vec3.x += velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.LEFT) vec3.x -= velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.UP) vec3.y += velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.DOWN) vec3.y -= velocity.maximum;

    // Update current velocity based on deltas of velocity and time:
    acceleration.current = acceleration.maximum * delta_time;
    velocity.x = approach(velocity.x, vec3.x, acceleration.current);
    velocity.y = approach(velocity.y, vec3.y, acceleration.current);
    velocity.z = approach(velocity.z, vec3.z, acceleration.current);

    // Compute movement delta (axis-aligned):
    vec3.x = velocity.x * delta_time;
    vec3.y = velocity.y * delta_time;
    vec3.z = velocity.z * delta_time;

    // Rotate movement delta:
    imul3D(&vec3, transform->yaw);

    // Apply movement delta to the current camera position:
    iadd3D(transform->position, &vec3);
}

void onMove2D(Transform2D* transform, f32 delta_time) {
    vec2.x = vec2.y = 0;

    // Compute velocity delta:
    if (keyboard.keys_pressed & keyboard.keys.FORWARD) vec3.y += velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.BACKWARD) vec3.y -= velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.RIGHT) vec3.x += velocity.maximum;
    if (keyboard.keys_pressed & keyboard.keys.LEFT) vec3.x -= velocity.maximum;

    // Update current velocity based on deltas of velocity and time:
    acceleration.current = acceleration.maximum * delta_time;
    velocity.x = approach(velocity.x, vec3.x, acceleration.current);
    velocity.y = approach(velocity.y, vec3.y, acceleration.current);

    // Compute movement delta (axis-aligned):
    vec3.x = velocity.x * delta_time;
    vec3.y = velocity.y * delta_time;

    // Rotate movement delta:
    imul2D(&vec2, transform->rotation);

    // Apply movement delta to the current camera position:
    iadd2D(transform->position, &vec2);
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