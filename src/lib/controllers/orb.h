#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

typedef struct {
    Controller controller;
    f32 pan_speed, dolly_speed, orbit_speed, dolly_amount, dolly_ratio, target_distance;
    Vector3 target_position, movement, scaled_right, scaled_up;
} OrbController;
OrbController orb = {
        .controller = {.type = CONTROLLER_ORB, .changed = {false, false, false}},
        .pan_speed = 1.0f / 100,
        .dolly_speed = 4,
        .orbit_speed = 1.0f / 1000,
        .dolly_amount = 0,
        .dolly_ratio = 4,
        .target_distance = 4
};

void onMouseScrolledOrb() {
    Transform3D* tr = orb.controller.camera->transform;

    scale3D(tr->forward, orb.dolly_ratio, &orb.movement);
    add3D(tr->position, &orb.movement, &orb.target_position);

    orb.dolly_amount += orb.dolly_speed * mouse.wheel.scroll;
    orb.dolly_ratio = orb.dolly_amount ? orb.target_distance * (
            orb.dolly_amount > 0 ? (
                    1 / orb.dolly_amount
            ) : (
                        1 - orb.dolly_amount
                ) / 2
    ) : orb.target_distance;

    scale3D(tr->forward, orb.dolly_ratio, &orb.movement);
    sub3D(&orb.target_position, &orb.movement, tr->position);

    orb.controller.changed.position = true;
}

void onMouseMovedOrb() {
    Transform3D* tr = orb.controller.camera->transform;

    if (mouse.buttons.right.is_down) {
        scale3D(tr->forward, orb.dolly_ratio, &orb.movement);
        add3D(tr->position, &orb.movement, &orb.target_position);

        yaw3D(orb.orbit_speed * (f32)-mouse.coords.relative.x, tr->yaw);
        pitch3D(orb.orbit_speed * (f32)-mouse.coords.relative.y, tr->pitch);
        matMul3D(tr->pitch, tr->yaw, tr->rotation);

        scale3D(tr->forward, orb.dolly_ratio, &orb.movement);
        sub3D(&orb.target_position, &orb.movement, tr->position);

        orb.controller.changed.orientation = true;
        orb.controller.changed.position = true;
    } else if (mouse.buttons.middle.is_down) {
        scale3D(tr->right, orb.pan_speed * (f32)-mouse.coords.relative.x, &orb.scaled_right);
        scale3D(tr->up, orb.pan_speed * (f32)+mouse.coords.relative.y, &orb.scaled_up);
        add3D(&orb.scaled_right, &orb.scaled_up, &orb.movement);
        iadd3D(tr->position, &orb.movement);

        orb.controller.changed.position = true;
    }
}

void onUpdateOrb() {}