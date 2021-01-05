#pragma once

#include "lib/core/types.h"
#include "lib/globals/app.h"
#include "lib/globals/camera.h"
#include "lib/math/math3D.h"
#include "lib/nodes/transform.h"
#include "lib/controllers/camera_controller.h"


void onMouseScrolledFps() {
    f32 zoom = fps_camera_controller.zoom_amount + mouse_wheel_scroll_amount * ZOOM_SPEED;
    fps_camera_controller.controller.camera->focal_length = zoom > 1 ? zoom : (zoom < -1 ? (-1/zoom) : 1);
    fps_camera_controller.zoom_amount = zoom;
    fps_camera_controller.controller.zoomed = true;
    mouse_wheel_scrolled = false;
    mouse_wheel_scroll_amount = 0;
}

void onMouseMovedFps() {
    rotateXform3(&fps_camera_controller.controller.camera->transform,
             MOUSE_TURN_SPEED * -mouse_pos_raw_diff.x,
             MOUSE_TURN_SPEED * -mouse_pos_raw_diff.y,
             0);
    fps_camera_controller.controller.turned = true;
    mouse_pos_raw_diff.x = 0;
    mouse_pos_raw_diff.y = 0;
    mouse_moved = false;
}

void onUpdateFps() {
    xform3 *xform = &fps_camera_controller.controller.camera->transform;
    vec3 *position = &xform->position;

    vec3 *movement = &fps_camera_controller.movement;
    vec3 *current_velocity = &fps_camera_controller.current_velocity;
    vec3 *target_velocity = &fps_camera_controller.target_velocity;

    fillVec3(target_velocity, 0);
    if (move_right)    target_velocity->x += MAX_VELOCITY;
    if (move_left)     target_velocity->x -= MAX_VELOCITY;
    if (move_up)       target_velocity->y += MAX_VELOCITY;
    if (move_down)     target_velocity->y -= MAX_VELOCITY;
    if (move_forward)  target_velocity->z += MAX_VELOCITY;
    if (move_backward) target_velocity->z -= MAX_VELOCITY;
    if (turn_right ||
        turn_left) {
        fps_camera_controller.controller.turned = true;
        f32 yaw = update_timer.delta_time * KEYBOARD_TURN_SPEED;
        rotateXform3(xform, turn_left ? yaw : -yaw, 0, 0);
    }

    // Update the current velocity:
    approachVec3(current_velocity, target_velocity, update_timer.delta_time * MAX_ACCELERATION);
    fps_camera_controller.controller.moved = nonZeroVec3(current_velocity);
    if (fps_camera_controller.controller.moved) { // Update the current position:
        *movement = *current_velocity;
        iscaleVec3(movement, update_timer.delta_time);
        imulVec3Mat3(movement, &xform->rotation_matrix);

        fps_camera_controller.old_position = *position;
        iaddVec3(position, movement);
    }
}

void initFpsController(Camera* camera) {
    initCameraController(camera,
            &fps_camera_controller.controller,
            CONTROLLER_FPS,
            onUpdateFps,
            onMouseMovedFps,
            onMouseScrolledFps);

    fillVec3(&fps_camera_controller.movement, 0);
    fillVec3(&fps_camera_controller.old_position, 0);
    fillVec3(&fps_camera_controller.target_velocity, 0);
    fillVec3(&fps_camera_controller.current_velocity, 0);

    fps_camera_controller.zoom_amount = camera->focal_length;
}