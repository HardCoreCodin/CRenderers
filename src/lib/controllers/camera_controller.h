#pragma once

#include "lib/core/types.h"
#include "lib/globals/camera.h"

void initCameraController(
        Camera *camera,
        CameraController *camera_controller,
        enum ControllerType type,
        CallBack on_update,
        CallBack on_mouse_moved,
        CallBack on_mouse_wheel_scrolled) {
    camera_controller->camera = camera;
    camera_controller->type = type;
    camera_controller->onUpdate = on_update;
    camera_controller->onMouseMoved = on_mouse_moved;
    camera_controller->onMouseWheelScrolled = on_mouse_wheel_scrolled;

    camera_controller->moved = false;
    camera_controller->turned = false;
    camera_controller->zoomed = false;
}