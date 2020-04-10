#pragma once

#include "lib/core/types.h"

#define MOUSE_CLICK_TICKS 10000

static Mouse mouse;

void initMouse() {
    mouse.is_captured = false;
    mouse.has_moved = false;
    mouse.double_clicked = false;

    mouse.wheel.was_scrolled = false;
    mouse.wheel.scroll_amount = 0;

    mouse.coords.absolute.x = 0;
    mouse.coords.absolute.y = 0;
    mouse.coords.relative.x = 0;
    mouse.coords.relative.y = 0;

    mouse.buttons.left.is_down = false;
    mouse.buttons.left.clicked = false;
    mouse.buttons.left.up.ticks = 0;
    mouse.buttons.left.down.ticks = 0;

    mouse.buttons.right.is_down = false;
    mouse.buttons.right.clicked = false;
    mouse.buttons.right.down.ticks = 0;
    mouse.buttons.right.up.ticks = 0;

    mouse.buttons.middle.is_down = false;
    mouse.buttons.middle.clicked = false;
    mouse.buttons.middle.down.ticks = 0;
    mouse.buttons.middle.up.ticks = 0;
}

void OnMouseMovedAbsolute(s16 x, s16 y) {
    mouse.has_moved = true;
    mouse.coords.absolute.x = x;
    mouse.coords.absolute.y = y;
}

void OnMouseMovedRelative(s16 dx, s16 dy) {
    mouse.has_moved = true;
    mouse.coords.relative.x += dx;
    mouse.coords.relative.y += dy;
}

void OnMouseWheelScrolled(f32 amount) {
    mouse.wheel.scroll_amount = amount;
    mouse.wheel.was_scrolled = true;
}

void OnMouseLeftButtonDown(s16 x, s16 y, u64 ticks) {
    mouse.buttons.left.down.ticks = ticks;
    mouse.buttons.left.down.coords.x = x;
    mouse.buttons.left.down.coords.y = y;
    mouse.buttons.left.is_down = true;
}

void OnMouseLeftButtonUp(s16 x, s16 y, u64 ticks) {
    mouse.buttons.left.up.ticks = ticks;
    mouse.buttons.left.up.coords.x = x;
    mouse.buttons.left.up.coords.y = y;
    mouse.buttons.left.is_down = false;
    mouse.buttons.left.clicked = ticks - mouse.buttons.left.down.ticks < MOUSE_CLICK_TICKS;
}

void OnMouseRightButtonDown(s16 x, s16 y, u64 ticks) {
    mouse.buttons.right.down.ticks = ticks;
    mouse.buttons.right.down.coords.x = x;
    mouse.buttons.right.down.coords.y = y;
    mouse.buttons.right.is_down = true;
}

void OnMouseRightButtonUp(s16 x, s16 y, u64 ticks) {
    mouse.buttons.right.up.ticks = ticks;
    mouse.buttons.right.up.coords.x = x;
    mouse.buttons.right.up.coords.y = y;
    mouse.buttons.right.is_down = false;
    mouse.buttons.right.clicked = ticks - mouse.buttons.right.down.ticks < MOUSE_CLICK_TICKS;
}

void OnMouseMiddleButtonDown(s16 x, s16 y, u64 ticks) {
    mouse.buttons.middle.down.ticks = ticks;
    mouse.buttons.middle.down.coords.x = x;
    mouse.buttons.middle.down.coords.y = y;
    mouse.buttons.middle.is_down = true;
}

void OnMouseMiddleButtonUp(s16 x, s16 y, u64 ticks) {
    mouse.buttons.middle.up.ticks = ticks;
    mouse.buttons.middle.up.coords.x = x;
    mouse.buttons.middle.up.coords.y = y;
    mouse.buttons.middle.is_down = false;
    mouse.buttons.middle.clicked = ticks - mouse.buttons.middle.down.ticks < MOUSE_CLICK_TICKS;
}