#pragma once

#include "lib/core/types.h"

typedef struct Keys {
    u8 FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN, HUD;
} Keys;

typedef struct Keyboard {
    u8 keys_pressed;
    Keys keys;
} Keyboard;
Keyboard keyboard;

void initKeyboard() {
    keyboard.keys_pressed = 0;

    keyboard.keys.FORWARD  = (u8)1 << (u8)0;
    keyboard.keys.BACKWARD = (u8)1 << (u8)1;
    keyboard.keys.LEFT     = (u8)1 << (u8)2;
    keyboard.keys.RIGHT    = (u8)1 << (u8)3;
    keyboard.keys.UP       = (u8)1 << (u8)4;
    keyboard.keys.DOWN     = (u8)1 << (u8)5;
    keyboard.keys.HUD      = (u8)1 << (u8)6;
}

void OnKeyDown(u8 key) {
    keyboard.keys_pressed |= key;
}

void OnKeyUp(u8 key) {
    keyboard.keys_pressed &= (u8)~key;
}