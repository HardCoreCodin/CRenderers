#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"

#define MOUSE_CLICK_TICKS 10000

Mouse* createMouse() {
    Mouse* mouse = Alloc(Mouse);
    mouse->is_captured = false;
    mouse->double_clicked = false;
    mouse->coords.absolute.changed = false;
    mouse->coords.relative.changed = false;

    mouse->wheel.changed = false;
    mouse->wheel.scroll = 0;

    mouse->coords.absolute.x = 0;
    mouse->coords.absolute.y = 0;
    mouse->coords.relative.x = 0;
    mouse->coords.relative.y = 0;

    mouse->buttons.left.is_down = false;
    mouse->buttons.left.clicked = false;
    mouse->buttons.left.up.ticks = 0;
    mouse->buttons.left.down.ticks = 0;

    mouse->buttons.right.is_down = false;
    mouse->buttons.right.clicked = false;
    mouse->buttons.right.down.ticks = 0;
    mouse->buttons.right.up.ticks = 0;

    mouse->buttons.middle.is_down = false;
    mouse->buttons.middle.clicked = false;
    mouse->buttons.middle.down.ticks = 0;
    mouse->buttons.middle.up.ticks = 0;

    return mouse;
}

void onMouseMovedAbsolute(Mouse* mouse, s16 x, s16 y) {
    mouse->coords.absolute.changed = true;
    mouse->coords.absolute.x = x;
    mouse->coords.absolute.y = y;
}

void onMouseMovedRelative(Mouse* mouse, s16 dx, s16 dy) {
    mouse->coords.relative.changed = true;
    mouse->coords.relative.x += dx;
    mouse->coords.relative.y += dy;
}

void onMouseWheelScrolled(Mouse* mouse, f32 amount) {
    mouse->wheel.scroll = amount;
    mouse->wheel.changed = true;
}

inline void onMouseButtonDown(MouseButton* mouse_button, s16 x, s16 y, u64 ticks) {
    mouse_button->down.ticks = ticks;
    mouse_button->down.coords.x = x;
    mouse_button->down.coords.y = y;
    mouse_button->is_down = true;
}

inline void onMouseButtonUp(MouseButton* mouse_button, s16 x, s16 y, u64 ticks) {
    mouse_button->up.ticks = ticks;
    mouse_button->up.coords.x = x;
    mouse_button->up.coords.y = y;
    mouse_button->is_down = false;
    mouse_button->clicked = ticks - mouse_button->down.ticks < MOUSE_CLICK_TICKS;
}

void onMouseLeftButtonDown(Mouse* mouse, s16 x, s16 y, u64 ticks ) { onMouseButtonDown(&mouse->buttons.left, x, y, ticks); }
void onMouseLeftButtonUp(Mouse* mouse, s16 x, s16 y, u64 ticks   ) { onMouseButtonUp(&mouse->buttons.left, x, y, ticks); }
void onMouseRightButtonDown(Mouse* mouse, s16 x, s16 y, u64 ticks ) { onMouseButtonDown(&mouse->buttons.right, x, y, ticks); }
void onMouseRightButtonUp(Mouse* mouse, s16 x, s16 y, u64 ticks   ) { onMouseButtonUp(&mouse->buttons.right, x, y, ticks); }
void onMouseMiddleButtonDown(Mouse* mouse, s16 x, s16 y, u64 ticks ) { onMouseButtonDown(&mouse->buttons.middle, x, y, ticks); }
void onMouseMiddleButtonUp(Mouse* mouse, s16 x, s16 y, u64 ticks   ) { onMouseButtonUp(&mouse->buttons.middle, x, y, ticks); }