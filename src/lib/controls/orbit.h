#pragma once

#include "lib/core/memory.h"
#include "lib/core/types.h"
#include "lib/nodes/transform.h"
#include "lib/nodes/camera.h"
#include "lib/math/math3D.h"

typedef struct Pan   {
    f32 sensitivity, right, up;
    bool changed;
} Pan;
void onPan(f32 right, f32 up, Pan* pan) {
    pan->right += right * pan->sensitivity;
    pan->up += up * pan->sensitivity;
    pan->changed = true;
}

typedef struct Dolly {
    f32 sensitivity, in, current, factor;
    bool changed;
} Dolly;
void onDolly(f32 in, Dolly* dolly) {
    dolly->in += in * dolly->sensitivity;
    dolly->changed = true;
}

typedef struct Orbit {
    f32 sensitivity, azimuth, elevation;
    bool changed;
} Orbit;
void onOrbit(f32 azimuth, f32 elevation, Orbit* orbit) {
    orbit->azimuth += azimuth * orbit->sensitivity;
    orbit->elevation += elevation * orbit->sensitivity;
    orbit->changed = true;
}

typedef struct OrbitController3D {
    Pan pan;
    Dolly dolly;
    Orbit orbit;
    Vector3* target_position;
    Vector3* target_direction;
} OrbitController3D;

void initOrbitController(OrbitController3D* orbit_controller) {
    orbit_controller->target_position = (Vector3*)allocate(sizeof(Vector3));
    orbit_controller->target_direction = (Vector3*)allocate(sizeof(Vector3));

    orbit_controller->pan.sensitivity = 1 / 100.0f;
    orbit_controller->pan.changed = false;

    orbit_controller->orbit.sensitivity = 1 / 1000.0f;

    orbit_controller->dolly.sensitivity = 1;
    orbit_controller->dolly.current = -5;
    orbit_controller->dolly.factor = -orbit_controller->dolly.current;
    orbit_controller->dolly.changed = false;
}

void dolly(Camera3D* camera, OrbitController3D* orbit_controller) {
    scale3D(camera->transform->forward, orbit_controller->dolly.factor, orbit_controller->target_direction);
    add3D(camera->position, orbit_controller->target_direction, orbit_controller->target_position);

    orbit_controller->dolly.current += orbit_controller->dolly.in;
    if (orbit_controller->dolly.current > 0)
        orbit_controller->dolly.factor = 1 / orbit_controller->dolly.current;
    else
        orbit_controller->dolly.factor = -orbit_controller->dolly.current;

    scale3D(camera->transform->forward, orbit_controller->dolly.factor, orbit_controller->target_direction);
    sub3D(orbit_controller->target_position, orbit_controller->target_direction, camera->position);

    orbit_controller->dolly.in = 0;
    orbit_controller->dolly.changed = false;
}

void pan(Camera3D* camera, OrbitController3D* orbit_controller) {
    scale3D(camera->transform->right, orbit_controller->pan.right, &vec3);
    iadd3D(camera->position, &vec3);

    scale3D(camera->transform->up, orbit_controller->pan.up, &vec3);
    iadd3D(camera->position, &vec3);

    orbit_controller->pan.right = orbit_controller->pan.up = 0;
    orbit_controller->pan.changed = false;
}

void orbit(Camera3D* camera, OrbitController3D* orbit_controller) {
    scale3D(camera->transform->forward, orbit_controller->dolly.factor, orbit_controller->target_direction);
    add3D(camera->position, orbit_controller->target_direction, orbit_controller->target_position);

    rotate3D(orbit_controller->orbit.azimuth,
             orbit_controller->orbit.elevation,
             0,
             camera->transform);

    scale3D(camera->transform->forward, orbit_controller->dolly.factor, orbit_controller->target_direction);
    sub3D(orbit_controller->target_position, orbit_controller->target_direction, camera->position);

    orbit_controller->orbit.azimuth = orbit_controller->orbit.elevation = 0;
    orbit_controller->orbit.changed = false;
}


