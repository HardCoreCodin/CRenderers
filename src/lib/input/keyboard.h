#pragma once

#include "lib/core/types.h"

typedef struct Button {
    u32 key;
    bool is_pressed;
} Button;

typedef struct Buttons {
    Button forward, back, left, right, up, down, hud;
} Buttons;
Buttons buttons;

void initButtons() {
    buttons.up.is_pressed = false;
    buttons.down.is_pressed = false;
    buttons.left.is_pressed = false;
    buttons.right.is_pressed = false;
    buttons.forward.is_pressed = false;
    buttons.back.is_pressed = false;
    buttons.hud.is_pressed = false;
};

//typedef struct Keys {
//    u8 FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN, HUD;
//} Keys;
//
//typedef struct Keyboard {
//    u8 keys_pressed;
//    Keys keys;
//} Keyboard;
//Keyboard keyboard;
//
//void initKeyboard() {
//    keyboard.keys_pressed = 0;
//
//    keyboard.keys.FORWARD  = (u8)1 << (u8)0;
//    keyboard.keys.BACKWARD = (u8)1 << (u8)1;
//    keyboard.keys.LEFT     = (u8)1 << (u8)2;
//    keyboard.keys.RIGHT    = (u8)1 << (u8)3;
//    keyboard.keys.UP       = (u8)1 << (u8)4;
//    keyboard.keys.DOWN     = (u8)1 << (u8)5;
//    keyboard.keys.HUD      = (u8)1 << (u8)6;
//}

void OnKeyDown(u32 key) {
    if (key == buttons.hud.key) buttons.hud.is_pressed = true;
    else if (key == buttons.up.key) buttons.up.is_pressed = true;
    else if (key == buttons.down.key) buttons.down.is_pressed = true;
    else if (key == buttons.left.key) buttons.left.is_pressed = true;
    else if (key == buttons.right.key) buttons.right.is_pressed = true;
    else if (key == buttons.forward.key) buttons.forward.is_pressed = true;
    else if (key == buttons.back.key) buttons.back.is_pressed = true;
}

void OnKeyUp(u8 key) {
    if (key == buttons.hud.key) buttons.hud.is_pressed = false;
    else if (key == buttons.up.key) buttons.up.is_pressed = false;
    else if (key == buttons.down.key) buttons.down.is_pressed = false;
    else if (key == buttons.left.key) buttons.left.is_pressed = false;
    else if (key == buttons.right.key) buttons.right.is_pressed = false;
    else if (key == buttons.forward.key) buttons.forward.is_pressed = false;
    else if (key == buttons.back.key) buttons.back.is_pressed = false;
}