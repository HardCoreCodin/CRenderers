#pragma once
#include "types.h"
#include "core.h"
#include "math3D.h"

typedef struct Camera {
    f32 focal_length;
    Vector3* position;
} Camera;

Camera camera = {1};
Vector3 delta;

Vector3* forward;
Vector3* right;
Vector3* up;

typedef struct Pan          {f32 sensitivity, right, up; bool changed;} Pan;
typedef struct Zoom         {f32 sensitivity, in; bool changed;} Zoom;
typedef struct Dolly        {f32 sensitivity, in, current, factor; bool changed;} Dolly;
typedef struct Orientation  {f32 sensitivity, yaw, pitch; bool changed;} Orientation;
typedef struct Velocity     {f32 x, y, z, maximum;} Velocity;
typedef struct Acceleration {f32 current, maximum;} Acceleration;

typedef struct FpsController {
    Zoom zoom;
    Velocity velocity;
    Orientation orientation;
    Acceleration acceleration;
} FpsController;
FpsController fps_controller;

typedef struct OrbitController {
    Pan pan;
    Dolly dolly;
    Orientation orbit;
    Vector3* target_position;
    Vector3* target_direction;
} OrbitController;
OrbitController orbit_controller;

void init_core3D() {
    up = &rotation_matrix.j;
    right = &rotation_matrix.i;
    forward = &rotation_matrix.k;
    orbit_controller.target_position = (Vector3*)allocate_memory(sizeof(Vector3));
    orbit_controller.target_direction = (Vector3*)allocate_memory(sizeof(Vector3));

    fps_controller.velocity.maximum = 8;
    fps_controller.acceleration.maximum = 10;
    fps_controller.orientation.sensitivity = 0.025f;
    fps_controller.zoom.sensitivity = 1;
    fps_controller.zoom.changed = fps_controller.orientation.changed = false;

    orbit_controller.pan.sensitivity = 0.2f;
    orbit_controller.orbit.sensitivity = 0.025f;
    orbit_controller.dolly.sensitivity = 1;
    orbit_controller.dolly.changed = orbit_controller.pan.changed = false;
    orbit_controller.dolly.current = -5;
    orbit_controller.dolly.factor = -orbit_controller.dolly.current;
}

void onMousePositionChanged(f32 dx, f32 dy) {
    if (mouse.is_captured) {
        fps_controller.orientation.yaw -= dx * fps_controller.orientation.sensitivity;
        fps_controller.orientation.pitch -= dy * fps_controller.orientation.sensitivity;
        fps_controller.orientation.changed = true;
    } else if (mouse.pressed) {
        if (mouse.pressed & MIDDLE) {
            orbit_controller.pan.right -= dx * orbit_controller.pan.sensitivity;
            orbit_controller.pan.up += dy * orbit_controller.pan.sensitivity;
            orbit_controller.pan.changed = true;
        } else if (mouse.pressed & RIGHT) {
            orbit_controller.orbit.yaw -= dx * orbit_controller.orbit.sensitivity;
            orbit_controller.orbit.pitch -= dy * orbit_controller.orbit.sensitivity;
            orbit_controller.orbit.changed = true;
        }
    }
}
void onMouseWheelChanged(f32 amount) {
    if (mouse.is_captured) {
        fps_controller.zoom.in += amount * fps_controller.zoom.sensitivity;
        fps_controller.zoom.changed = true;
    } else {
        orbit_controller.dolly.in += amount * orbit_controller.dolly.sensitivity;
        orbit_controller.dolly.changed = true;
    }
}

bool dolly(f32 delta_time) {
    if (!orbit_controller.dolly.changed)
        return false;

    scale3D(forward, orbit_controller.dolly.factor, orbit_controller.target_direction);
    add3D(camera.position, orbit_controller.target_direction, orbit_controller.target_position);

    orbit_controller.dolly.current += orbit_controller.dolly.in * delta_time;
    if (orbit_controller.dolly.current > 0)
        orbit_controller.dolly.factor = 1 / orbit_controller.dolly.current;
    else
        orbit_controller.dolly.factor = -orbit_controller.dolly.current;

    scale3D(forward, orbit_controller.dolly.factor, orbit_controller.target_direction);
    sub3D(orbit_controller.target_position, orbit_controller.target_direction, camera.position);

    orbit_controller.dolly.in = 0;
    orbit_controller.dolly.changed = false;

    return true;
}

bool pan(f32 delta_time) {
    if (!orbit_controller.pan.changed)
        return false;

    scale3D(right, orbit_controller.pan.right * delta_time, &delta);
    iadd3D(camera.position, &delta);

    scale3D(up, orbit_controller.pan.up * delta_time, &delta);
    iadd3D(camera.position, &delta);

    orbit_controller.pan.right = orbit_controller.pan.up = 0;
    orbit_controller.pan.changed = false;

    return true;
}

bool orbit(f32 delta_time) {
    if (!orbit_controller.orbit.changed)
        return false;

    scale3D(forward, orbit_controller.dolly.factor, orbit_controller.target_direction);
    add3D(camera.position, orbit_controller.target_direction, orbit_controller.target_position);

    rotate(orbit_controller.orbit.yaw * delta_time,
           orbit_controller.orbit.pitch * delta_time,
           0);

    scale3D(forward, orbit_controller.dolly.factor, orbit_controller.target_direction);
    sub3D(orbit_controller.target_position, orbit_controller.target_direction, camera.position);

    orbit_controller.orbit.yaw = orbit_controller.orbit.pitch = 0;
    orbit_controller.orbit.changed = false;

    return true;
}

bool zoom(f32 delta_time) {
    if (!fps_controller.zoom.changed)
        return false;

    camera.focal_length += fps_controller.zoom.in * delta_time;

    fps_controller.zoom.in = 0;
    fps_controller.zoom.changed = false;

    return true;
}

bool look(f32 delta_time) {
    if (!fps_controller.orientation.changed)
        return false;

    rotate(fps_controller.orientation.yaw * delta_time,
           fps_controller.orientation.pitch * delta_time,
           0);

    fps_controller.orientation.yaw = 0;
    fps_controller.orientation.pitch = 0;
    fps_controller.orientation.changed = false;

    return true;
}

void move(f32 delta_time) {
    delta.x = delta.y = delta.z = 0;

    // Compute velocity delta:
    if (keyboard.pressed & FORWARD) delta.z += fps_controller.velocity.maximum;
    if (keyboard.pressed & BACKWARD) delta.z -= fps_controller.velocity.maximum;
    if (keyboard.pressed & RIGHT) delta.x += fps_controller.velocity.maximum;
    if (keyboard.pressed & LEFT) delta.x -= fps_controller.velocity.maximum;
    if (keyboard.pressed & UP) delta.y += fps_controller.velocity.maximum;
    if (keyboard.pressed & DOWN) delta.y -= fps_controller.velocity.maximum;

    // Update current velocity based on deltas of velocity and time:
    fps_controller.acceleration.current = fps_controller.acceleration.maximum * delta_time;
    fps_controller.velocity.x = approach(fps_controller.velocity.x, delta.x, fps_controller.acceleration.current);
    fps_controller.velocity.y = approach(fps_controller.velocity.y, delta.y, fps_controller.acceleration.current);
    fps_controller.velocity.z = approach(fps_controller.velocity.z, delta.z, fps_controller.acceleration.current);

    // Compute movement delta (axis-aligned):
    delta.x = fps_controller.velocity.x * delta_time;
    delta.y = fps_controller.velocity.y * delta_time;
    delta.z = fps_controller.velocity.z * delta_time;

    // Rotate movement delta:
    imul3D(&delta, &yaw_matrix);

    // Apply movement delta to the current camera position:
    iadd3D(camera.position, &delta);
}