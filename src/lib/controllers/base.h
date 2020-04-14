#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"

void updateController(Controller* controller) {
    if (mouse.wheel.changed) {
        mouse.wheel.changed = false;
        controller->on.mouse_scrolled();
        mouse.wheel.scroll = 0;
    }

    if (mouse.coords.relative.changed) {
        mouse.coords.relative.changed = false;
        controller->on.mouse_moved();
        mouse.coords.relative.x = 0;
        mouse.coords.relative.y = 0;
    }

    if (controller->changed.fov) {
        controller->changed.fov = false;
        controller->zoom(controller);
    }

    if (controller->changed.orientation) {
        controller->changed.orientation = false;
        controller->rotate(controller);
    }

    if (controller->changed.position) {
        controller->changed.position = false;
        controller->move(controller);
    }

    controller->on.update();
}

void initController(
        Controller* controller,
        Camera* camera,
        Callback update,
        Callback mouse_moved,
        Callback mouse_scrolled) {
    controller->camera = camera;
    controller->changed.orientation = false;
    controller->changed.position = false;
    controller->changed.fov = false;

    controller->on.update = update;
    controller->on.mouse_moved = mouse_moved;
    controller->on.mouse_scrolled = mouse_scrolled;
}