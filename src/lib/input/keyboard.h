#pragma once

#include "lib/core/types.h"

bool is_running = true;
enum ShadingMode shading_mode = Normal;

bool move_right,
     turn_right,
     move_left, turn_left,
     move_up,
     move_down,
     move_forward,
     move_backward,
     ctrl_is_pressed,
     show_hud;

u8 up_key,
   down_key,
   left_key, turn_left_key,
   right_key, turn_right_key,
   forward_key,
   backward_key,
   toggle_hud_key,
   normal_key,
   lambert_key,
   phong_key,
   blinn_key,
   ctrl_key,
   exit_key;

void keyChanged(u8 key, bool pressed) {
    if      (key == turn_left_key) turn_left = pressed;
    else if (key == turn_right_key) turn_right = pressed;
    else if (key == left_key) move_left = pressed;
    else if (key == right_key) move_right = pressed;
    else if (key == up_key) move_up = pressed;
    else if (key == down_key) move_down = pressed;
    else if (key == forward_key) move_forward = pressed;
    else if (key == backward_key) move_backward = pressed;
    else if (key == ctrl_key) ctrl_is_pressed = pressed;
    else if (key == normal_key) shading_mode = Normal;
    else if (key == lambert_key) shading_mode = Lambert;
    else if (key == phong_key) shading_mode = Phong;
    else if (key == blinn_key) shading_mode = Blinn;
    else if (key == exit_key) is_running = false;
    else if (key == toggle_hud_key && !pressed) show_hud = !show_hud;
}