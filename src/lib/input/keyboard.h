#pragma once

#include "lib/core/types.h"

Keyboard* createKeyboard() {
    Keyboard* keyboard = Alloc(Keyboard);

    keyboard->up.is_pressed = false;
    keyboard->down.is_pressed = false;
    keyboard->left.is_pressed = false;
    keyboard->right.is_pressed = false;
    keyboard->forward.is_pressed = false;
    keyboard->back.is_pressed = false;
    keyboard->hud.is_pressed = false;
    keyboard->first.is_pressed = false;
    keyboard->second.is_pressed = false;

    return keyboard;
};

void onKeyDown(Keyboard* keyboard, u32 key) {
    if (key == keyboard->hud.key)
        keyboard->hud.is_pressed = true;
    else if (key == keyboard->up.key)
        keyboard->up.is_pressed = true;
    else if (key == keyboard->down.key)
        keyboard->down.is_pressed = true;
    else if (key == keyboard->left.key)
        keyboard->left.is_pressed = true;
    else if (key == keyboard->right.key)
        keyboard->right.is_pressed = true;
    else if (key == keyboard->forward.key)
        keyboard->forward.is_pressed = true;
    else if (key == keyboard->back.key)
        keyboard->back.is_pressed = true;
    else if (key == keyboard->first.key)
        keyboard->first.is_pressed = true;
    else if (key == keyboard->second.key)
        keyboard->second.is_pressed = true;
}

void onKeyUp(Keyboard* keyboard, u8 key) {
    if (key == keyboard->hud.key)
        keyboard->hud.is_pressed = false;
    else if (key == keyboard->up.key)
        keyboard->up.is_pressed = false;
    else if (key == keyboard->down.key)
        keyboard->down.is_pressed = false;
    else if (key == keyboard->left.key)
        keyboard->left.is_pressed = false;
    else if (key == keyboard->right.key)
        keyboard->right.is_pressed = false;
    else if (key == keyboard->forward.key)
        keyboard->forward.is_pressed = false;
    else if (key == keyboard->back.key)
        keyboard->back.is_pressed = false;
    else if (key == keyboard->first.key)
        keyboard->first.is_pressed = false;
    else if (key == keyboard->second.key)
        keyboard->second.is_pressed = false;
}