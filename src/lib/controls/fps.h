#pragma once

#include "lib/core/types.h"
#include "lib/math/math1D.h"
#include "lib/nodes/transform.h"
#include "lib/controls/controls.h"

typedef struct Zoom {
    f32 sensitivity, in;
    bool changed;
} Zoom;
Zoom zoom = {1, 0, false};

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

void onZoom(f32 in) {
    zoom.in += in * zoom.sensitivity;
    zoom.changed = true;
}

void onOrient(f32 yaw, f32 pitch) {
    orientation.yaw += yaw * orientation.sensitivity;
    orientation.pitch += pitch * orientation.sensitivity;
    orientation.changed = true;
}

void updateZoom(f32* focal_length) {
    *focal_length += zoom.in;

    zoom.in = 0;
    zoom.changed = false;
}

void updateOrientation(Transform3D* transform) {
    rotate3D(orientation.yaw, orientation.pitch,0, transform);

    orientation.yaw = 0;
    orientation.pitch = 0;
    orientation.changed = false;
}

void updatePosition(Transform3D* transform, f32 delta_time) {
    vec3.x = vec3.y = vec3.z = 0;

    // Compute velocity delta:
    if (controls.keyboard.pressed & controls.buttons.FORWARD) vec3.z += velocity.maximum;
    if (controls.keyboard.pressed & controls.buttons.BACKWARD) vec3.z -= velocity.maximum;
    if (controls.keyboard.pressed & controls.buttons.RIGHT) vec3.x += velocity.maximum;
    if (controls.keyboard.pressed & controls.buttons.LEFT) vec3.x -= velocity.maximum;
    if (controls.keyboard.pressed & controls.buttons.UP) vec3.y += velocity.maximum;
    if (controls.keyboard.pressed & controls.buttons.DOWN) vec3.y -= velocity.maximum;

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