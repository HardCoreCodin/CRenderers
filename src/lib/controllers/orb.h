#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/timers.h"
#include "lib/globals/camera.h"


void onMouseScrolledOrb() { // Dolly
    xform3 *xform = &orb_camera_controller.controller.camera->transform;
    vec3 *forward = xform->forward_direction;
    vec3 *position = &xform->position;
    vec3 *movement = &orb_camera_controller.movement;
    vec3 *target_position = &orb_camera_controller.target_position;
    f32 target_distance = orb_camera_controller.target_distance;
    f32 dolly = orb_camera_controller.dolly_amount;

    // Compute target position:
    scaleVec3(forward, target_distance, movement);
    addVec3(position, movement, target_position);

    // Compute new target distance:
    dolly += mouse_wheel_scroll_amount * DOLLY_SPEED;
    target_distance = powf(2, dolly / -200) * ORBIT_TARGET_DISTANCE;

    // Back-track from target position to new current position:
    scaleVec3(forward, target_distance, movement);
    subVec3(target_position, movement, position);

    mouse_wheel_scroll_amount = 0;
    mouse_wheel_scrolled = false;
    orb_camera_controller.controller.moved = true;
    orb_camera_controller.target_distance = target_distance;
    orb_camera_controller.dolly_amount = dolly;
}

void onMouseMovedOrb() {
    xform3 *xform = &orb_camera_controller.controller.camera->transform;
    vec3 *forward = xform->forward_direction;
    vec3 *position = &xform->position;
    vec3 *movement = &orb_camera_controller.movement;
    vec3 *target_position = &orb_camera_controller.target_position;

    if (right_mouse_button.is_pressed) { // Orbit
        f32 target_distance = orb_camera_controller.target_distance;

        // Compute target position:
        scaleVec3(forward, target_distance, movement);
        addVec3(position, movement, target_position);

        // Compute new orientation at target position:
        rotateXform3(xform,
                -mouse_pos_diff.x * ORBIT_SPEED,
                -mouse_pos_diff.y * ORBIT_SPEED,
                0);

        // Back-track from target position to new current position:
        scaleVec3(forward, target_distance, movement);
        subVec3(target_position, movement, position);

        orb_camera_controller.controller.turned = true;
        orb_camera_controller.controller.moved = true;
    } else if (middle_mouse_button.is_pressed) { // Pan
        // Computed scaled up & right vectors:
        scaleVec3(xform->right_direction, -mouse_pos_diff.x * PAN_SPEED, &orb_camera_controller.scaled_right);
        scaleVec3(xform->up_direction, mouse_pos_diff.y * PAN_SPEED, &orb_camera_controller.scaled_up);

        // Move current position by the combined movement:
        addVec3(&orb_camera_controller.scaled_right, &orb_camera_controller.scaled_up, movement);
        iaddVec3(position, movement);

        orb_camera_controller.controller.moved = true;
    }

    mouse_moved = false;
    mouse_pos_diff.x = 0;
    mouse_pos_diff.y = 0;
}

void onUpdateOrb() {}

void initOrbController(Camera* camera) {
    initCameraController(camera,
                         &orb_camera_controller.controller,
                         CONTROLLER_ORB,
                         onUpdateOrb,
                         onMouseMovedOrb,
                         onMouseScrolledOrb);

    fillVec3(&orb_camera_controller.movement, 0);
    fillVec3(&orb_camera_controller.target_position, 0);
    fillVec3(&orb_camera_controller.scaled_right, 0);
    fillVec3(&orb_camera_controller.scaled_up, 0);

    orb_camera_controller.dolly_amount = 0;
    orb_camera_controller.target_distance = ORBIT_TARGET_DISTANCE;
}