#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/memory/allocators.h"
#include "lib/controllers/base.h"

typedef struct {
    f32 pan_speed, dolly_speed, orbit_speed, dolly_amount, dolly_ratio, target_distance;
    Vector3 target_position;
    Controller controller;
} OrbController;

OrbController orb = {
        .pan_speed = 1.0f / 100,
        .dolly_speed = 4,
        .orbit_speed = 1.0f / 1000,
        .dolly_amount = 0,
        .dolly_ratio = 4,
        .target_distance = 4
};

#define ORB_UPDATE_TARGET_POSITION \
Vector3* forward = orb.controller.camera->transform->forward; \
Vector3* position = orb.controller.camera->transform->position; \
f32 fx = forward->x; \
f32 fy = forward->y; \
f32 fz = forward->z; \
f32* px = &position->x; \
f32* py = &position->y; \
f32* pz = &position->z; \
orb.target_position.x = *px + fx*orb.dolly_ratio; \
orb.target_position.y = *py + fy*orb.dolly_ratio; \
orb.target_position.z = *pz + fz*orb.dolly_ratio

#define ORB_UPDATE_CURRENT_POSITION \
*px = orb.target_position.x - orb.dolly_ratio * fx; \
*py = orb.target_position.y - orb.dolly_ratio * fy; \
*pz = orb.target_position.z - orb.dolly_ratio * fz

inline void OrbDolly() {
    ORB_UPDATE_TARGET_POSITION;
    orb.dolly_amount += orb.dolly_speed * mouse.wheel.scroll;
    orb.dolly_ratio = orb.dolly_amount ? orb.target_distance * (
            orb.dolly_amount > 0 ? (
                1 / orb.dolly_amount
        ) : (
                1 - orb.dolly_amount
        ) / 2
    ) : orb.target_distance;
    ORB_UPDATE_CURRENT_POSITION;
}
inline void OrbPan() {
    f32 right = orb.pan_speed * (f32)-mouse.coords.relative.x;
    f32 up    = orb.pan_speed * (f32)+mouse.coords.relative.y;

    orb.controller.camera->transform->position->x += (
    orb.controller.camera->transform->right->x * right +
    orb.controller.camera->transform->up->x * up
    );
    orb.controller.camera->transform->position->y += (
    orb.controller.camera->transform->right->y * right +
    orb.controller.camera->transform->up->y * up
    );
    orb.controller.camera->transform->position->z += (
    orb.controller.camera->transform->right->z * right +
    orb.controller.camera->transform->up->z * up
    );
}
inline void OrbOrbit() {
    ORB_UPDATE_TARGET_POSITION;
    f32 yaw   = orb.orbit_speed * (f32)-mouse.coords.relative.x;
    f32 pitch = orb.orbit_speed * (f32)-mouse.coords.relative.y;
    if (yaw) yaw3D(yaw, orb.controller.camera->transform->yaw);
    if (pitch) pitch3D(pitch, orb.controller.camera->transform->pitch);
    matMul3D(orb.controller.camera->transform->pitch,
             orb.controller.camera->transform->yaw,
             orb.controller.camera->transform->rotation);
    ORB_UPDATE_CURRENT_POSITION;
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