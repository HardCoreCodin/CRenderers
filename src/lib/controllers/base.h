#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"

void initController(
        Controller* controller,
        Callback onUpdate,
        Callback onMouseMoved,
        Callback onMouseScrolled,
        f32 rotation_speed,
        f32 scroll_speed) {

    controller->on.update = onUpdate;
    controller->on.mouse_moved = onMouseMoved;
    controller->on.mouse_scrolled = onMouseScrolled;
    controller->mouse_scroll_speed = scroll_speed;
    controller->mouse_movement_speed = rotation_speed;
    controller->zoomed = controller->rotated = controller->moved = false;
    controller->movement = (Vector3*)allocate(sizeof(Vector3));
    controller->movement->x = controller->movement->y = controller->movement->z = 0;
}