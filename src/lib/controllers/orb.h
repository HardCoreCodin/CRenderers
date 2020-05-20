#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

OrbController* createOrbController(Camera* camera) {
    OrbController* orb_controller = Alloc(OrbController);
    orb_controller->controller.type = CONTROLLER_ORB;
    orb_controller->controller.camera = camera;

    orb_controller->target_position = Alloc(Vector3);
    orb_controller->scaled_right = Alloc(Vector3);
    orb_controller->scaled_up = Alloc(Vector3);
    orb_controller->movement = Alloc(Vector3);

    orb_controller->pan_speed = 1.0f / 100;
    orb_controller->dolly_speed = 4;
    orb_controller->orbit_speed = 1.0f / 1000;
    orb_controller->dolly_amount = 0;
    orb_controller->dolly_ratio = 4;
    orb_controller->target_distance = 4;

    return orb_controller;
}

void onMouseScrolledOrb(OrbController* orb, Mouse* mouse) {
    Transform3D* tr = orb->controller.camera->transform;
    Vector3* trg_p = orb->target_position;
    Vector3* mov = orb->movement;
    Vector3* pos = tr->position;
    Vector3* fwd = tr->forward;
    f32* ratio = &orb->dolly_ratio;

    // Compute target position:
    scale3D(fwd, *ratio, mov);
    add3D(pos, mov, trg_p);

    // Compute new ratio:
    orb->dolly_amount += orb->dolly_speed * mouse->wheel.scroll;
    *ratio = orb->dolly_amount ? orb->target_distance * (
            orb->dolly_amount > 0 ? (
                    1 / orb->dolly_amount
            ) : (
                        1 - orb->dolly_amount
                ) / 2
    ) : orb->target_distance;

    // Back-track from target position to new current position:
    scale3D(fwd, *ratio, mov);
    sub3D(trg_p, mov, pos);

    orb->controller.changed.position = true;
}

void onMouseMovedOrb(OrbController* orb, Mouse* mouse) {
    Transform3D* tr = orb->controller.camera->transform;
    Vector3* pos = tr->position;
    Vector3* mov = orb->movement;

    const f32 dx = (f32)mouse->coords.relative.x;
    const f32 dy = (f32)mouse->coords.relative.y;

    if (mouse->buttons.right.is_down) { // Orbit
        Vector3* trg_pos = orb->target_position;
        Vector3* fwd = tr->forward;
        const f32 ratio = orb->dolly_ratio;
        const f32 speed = orb->orbit_speed;

        // Compute target position:
        scale3D(fwd, ratio, mov);
        add3D(pos, mov, trg_pos);

        // Compute new orientation at target position:
        yaw3D(speed * -dx, tr->yaw);
        pitch3D(speed * -dy, tr->pitch);
        matMul3D(tr->pitch, tr->yaw, tr->rotation);

        // Back-track from target position to new current position:
        scale3D(fwd, ratio, mov);
        sub3D(trg_pos, mov, pos);

        orb->controller.changed.orientation = true;
        orb->controller.changed.position = true;
    } else if (mouse->buttons.middle.is_down) { // Pan
        Vector3* up = orb->scaled_up;
        Vector3* right = orb->scaled_right;
        const f32 speed = orb->pan_speed;

        // Computed scaled up & right vectors:
        scale3D(tr->right, speed * -dx, right);
        scale3D(tr->up, speed * dy, up);

        // Move current position by the combined movement:
        add3D(right, up, mov);
        iadd3D(pos, mov);

        orb->controller.changed.position = true;
    }
}

void onUpdateOrb(OrbController* orb, Keyboard* keyboard, f32 seconds_passed) {}