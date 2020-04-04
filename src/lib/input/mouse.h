#pragma once

#include "lib/core/types.h"

#define MOUSE_CLICK_TICKS 10000

typedef struct MouseCoords {
    Coords absolute, relative;
} MouseCoords;

typedef struct MouseButtonState {
    u64 ticks;
    Coords coords;
} MouseButtonState;

typedef struct MouseButton {
    bool is_down, clicked;
    MouseButtonState up, down;
} MouseButton;

typedef struct MouseButtons {
    MouseButton left, right, middle;
} MouseButtons;

typedef struct MouseWheel {
    bool was_scrolled;
    f32 scroll_amount;
} MouseWheel;

typedef struct MouseDoubleClick {
    bool was_actioned;
    Coords at;
} MouseDoubleClick;

typedef struct Mouse {
    bool is_captured, has_moved;
    MouseWheel wheel;
    MouseCoords coords;
    MouseButtons buttons;
    MouseDoubleClick double_click;
} Mouse;
Mouse mouse;

void initMouse() {
    mouse.is_captured = false;
    mouse.has_moved = false;

    mouse.double_click.was_actioned = false;
    mouse.double_click.at.x = 0;
    mouse.double_click.at.y = 0;

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

void OnMouseDoubleClicked(s16 x, s16 y) {
    mouse.double_click.at.x = x;
    mouse.double_click.at.y = y;
    mouse.double_click.was_actioned = true;
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