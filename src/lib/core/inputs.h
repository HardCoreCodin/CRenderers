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

typedef struct Input {
    Keyboard keyboard;
    Mouse mouse;
    Buttons buttons;
} Input;
static Input input;

void initInput() {
    input.mouse.is_captured = false;
    input.mouse.pressed = 0;

    input.keyboard.pressed = 0;

    input.buttons.FORWARD  = (u8)1 << (u8)0;
    input.buttons.BACKWARD = (u8)1 << (u8)1;
    input.buttons.LEFT     = (u8)1 << (u8)2;
    input.buttons.RIGHT    = (u8)1 << (u8)3;
    input.buttons.UP       = (u8)1 << (u8)4;
    input.buttons.DOWN     = (u8)1 << (u8)5;
    input.buttons.MIDDLE   = (u8)1 << (u8)6;
}