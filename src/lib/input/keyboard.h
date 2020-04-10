#pragma once

#include "lib/core/types.h"

static Buttons buttons;

void initButtons() {
    buttons.up.is_pressed = false;
    buttons.down.is_pressed = false;
    buttons.left.is_pressed = false;
    buttons.right.is_pressed = false;
    buttons.forward.is_pressed = false;
    buttons.back.is_pressed = false;
    buttons.hud.is_pressed = false;
    buttons.first.is_pressed = false;
    buttons.second.is_pressed = false;
};

void OnKeyDown(u32 key) {
    if (key == buttons.hud.key) buttons.hud.is_pressed = true;
    else if (key == buttons.up.key) buttons.up.is_pressed = true;
    else if (key == buttons.down.key) buttons.down.is_pressed = true;
    else if (key == buttons.left.key) buttons.left.is_pressed = true;
    else if (key == buttons.right.key) buttons.right.is_pressed = true;
    else if (key == buttons.forward.key) buttons.forward.is_pressed = true;
    else if (key == buttons.back.key) buttons.back.is_pressed = true;
    else if (key == buttons.first.key) buttons.first.is_pressed = true;
    else if (key == buttons.second.key) buttons.second.is_pressed = true;
}

void OnKeyUp(u8 key) {
    if (key == buttons.hud.key) buttons.hud.is_pressed = false;
    else if (key == buttons.up.key) buttons.up.is_pressed = false;
    else if (key == buttons.down.key) buttons.down.is_pressed = false;
    else if (key == buttons.left.key) buttons.left.is_pressed = false;
    else if (key == buttons.right.key) buttons.right.is_pressed = false;
    else if (key == buttons.forward.key) buttons.forward.is_pressed = false;
    else if (key == buttons.back.key) buttons.back.is_pressed = false;
    else if (key == buttons.first.key) buttons.first.is_pressed = false;
    else if (key == buttons.second.key) buttons.second.is_pressed = false;
}