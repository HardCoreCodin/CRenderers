#pragma once

#include "lib/core/types.h"

typedef struct Buttons {
    u8 FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN, MIDDLE;
} Buttons;

typedef struct Keyboard {
    u8 pressed;
} Keyboard;

typedef struct Mouse {
    u8 pressed, is_captured;
} Mouse;

typedef struct Controls {
    Keyboard keyboard;
    Mouse mouse;
    Buttons buttons;
} Controls;

void initControls(Controls* controls) {
    controls->mouse.is_captured = false;
    controls->mouse.pressed = 0;

    controls->keyboard.pressed = 0;

    controls->buttons.FORWARD  = (u8)1 << (u8)0;
    controls->buttons.BACKWARD = (u8)1 << (u8)1;
    controls->buttons.LEFT     = (u8)1 << (u8)2;
    controls->buttons.RIGHT    = (u8)1 << (u8)3;
    controls->buttons.UP       = (u8)1 << (u8)4;
    controls->buttons.DOWN     = (u8)1 << (u8)5;
    controls->buttons.MIDDLE   = (u8)1 << (u8)6;
}