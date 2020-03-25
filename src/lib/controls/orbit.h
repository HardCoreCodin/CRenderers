#pragma once

#include "lib/core/memory.h"
#include "lib/core/types.h"
#include "lib/nodes/transform.h"
#include "lib/nodes/camera.h"
#include "lib/math/math3D.h"

Vector3 target_position, target_direction;

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

void onPan(f32 right, f32 up) {
    pan.right += right * pan.sensitivity;
    pan.up += up * pan.sensitivity;
    pan.changed = true;
}

void onDolly(f32 in) {
    dolly.in += in * dolly.sensitivity;
    dolly.changed = true;
}


void onOrbit(f32 azimuth, f32 elevation) {
    orbit.azimuth += azimuth * orbit.sensitivity;
    orbit.elevation += elevation * orbit.sensitivity;
    orbit.changed = true;
}

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

void updatePan(Transform3D* transform) {
    scale3D(transform->right, pan.right, &vec3);
    iadd3D(transform->position, &vec3);

    scale3D(transform->up, pan.up, &vec3);
    iadd3D(transform->position, &vec3);

    pan.right = pan.up = 0;
    pan.changed = false;
}

void updateOrbit(Transform3D* transform) {
    scale3D(transform->forward, dolly.factor, &target_direction);
    add3D(transform->position, &target_direction, &target_position);

    rotate3D(orbit.azimuth, orbit.elevation,0,transform);

    scale3D(transform->forward, dolly.factor, &target_direction);
    sub3D(&target_position, &target_direction, transform->position);

    orbit.azimuth = orbit.elevation = 0;
    orbit.changed = false;
}


