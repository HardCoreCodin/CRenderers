#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/memory/allocators.h"
#include "lib/controllers/base.h"

typedef struct { f32 amount, ratio; } OrbControllerDolly;
typedef struct { f32 pan, dolly, orbit; } OrbControllerSpeed;
typedef struct {
    OrbControllerSpeed speed;
    OrbControllerDolly dolly;
    Controller controller;
    Matrix3x3 yaw, pitch, rotation;
    Vector3 *up, *right, *forward, *pan, *C, *O, *T, *O2T, *C2T;
    f32 target_distance;
} OrbController;

OrbController orb = {1.0f / 100, 1.0f / 10, 1.0f / 1000};

inline void OrbUpdateO2T() { scale3D(orb.forward, orb.target_distance, orb.O2T); }
inline void OrbUpdateC2T() { scale3D(orb.O2T, orb.dolly.ratio, orb.C2T); }
inline void OrbUpdateT() { add3D(orb.O, orb.O2T, orb.T); }
inline void OrbUpdateC() { sub3D(orb.T, orb.C2T, orb.C); }
inline void OrbUpdateO() { sub3D(orb.T, orb.O2T, orb.O); }
inline void OrbDolly() {
    orb.dolly.amount += orb.speed.dolly * mouse.wheel.scroll;
    orb.dolly.ratio = orb.dolly.amount > 0 ? 1 / orb.dolly.amount : -orb.dolly.amount;

    OrbUpdateC2T();
    OrbUpdateC();
}
inline void OrbPan() {
    f32 right = orb.speed.pan * (f32)-mouse.coords.relative.x;
    f32 up    = orb.speed.pan * (f32)+mouse.coords.relative.y;

    scale3D(orb.right, right, orb.pan);
    iadd3D(orb.C, orb.pan);
    iadd3D(orb.O, orb.pan);
    iadd3D(orb.T, orb.pan);

    scale3D(orb.up, up, orb.pan);
    iadd3D(orb.C, orb.pan);
    iadd3D(orb.O, orb.pan);
    iadd3D(orb.T, orb.pan);
}
inline void OrbOrbit() {
    f32 yaw   = orb.speed.orbit * (f32)-mouse.coords.relative.x;
    f32 pitch = orb.speed.orbit * (f32)-mouse.coords.relative.y;
    if (yaw) yaw3D(yaw, orb.yaw);
    if (pitch) pitch3D(pitch, orb.pitch);
    matMul3D(orb.pitch, orb.yaw, orb.rotation);

    OrbUpdateO2T();
    OrbUpdateO();
    OrbUpdateC2T();
    OrbUpdateC();
}

void OrbOnMouseScrolled() {
    OrbDolly();
    orb.controller.changed.position = true;
}

void OrbOnMouseMoved() {
    if (mouse.buttons.right.is_down) {
        OrbOrbit();
        orb.controller.changed.orientation = true;
        orb.controller.changed.position = true;
    } else if (mouse.buttons.middle.is_down) {
        OrbPan();
        orb.controller.changed.position = true;
    }
}

void OrbOnUpdate() {}

void OrbOnReset() {
    orb.dolly.amount = 1;
    orb.dolly.ratio = 1;

    orb.yaw = orb.controller.camera->transform->yaw;
    orb.pitch = orb.controller.camera->transform->pitch;
    orb.rotation = orb.controller.camera->transform->rotation;

    orb.up = orb.controller.camera->transform->up;
    orb.right = orb.controller.camera->transform->right;
    orb.forward = orb.controller.camera->transform->forward;

    orb.C = orb.controller.camera->transform->position;
    *orb.O = *orb.C;

    OrbUpdateO2T();
    OrbUpdateT();
}

void initOrbController(Camera* camera) {
    orb.target_distance = 10;

    orb.pan = Alloc(Vector3);
    orb.O = Alloc(Vector3);
    orb.T = Alloc(Vector3);
    orb.O2T = Alloc(Vector3);
    orb.C2T = Alloc(Vector3);

    initController(&orb.controller, camera, OrbOnReset, OrbOnUpdate, OrbOnMouseMoved, OrbOnMouseScrolled);
}

