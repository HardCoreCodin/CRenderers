#pragma once

#include "lib/core/types.h"

typedef struct {
    bool is_pressed,
         is_released;

    vec2i down_pos,
          up_pos;
} MouseButton;
MouseButton middle_mouse_button,
            right_mouse_button,
            left_mouse_button;

vec2i mouse_pos,
      mouse_pos_diff;

bool mouse_moved,
     mouse_is_captured,
     mouse_double_clicked,
     mouse_wheel_scrolled;

f32 mouse_wheel_scroll_amount;

bool move_right,
     turn_right,
     move_left, turn_left,
     move_up,
     move_down,
     move_forward,
     move_backward,
     alt_is_pressed,
     ctrl_is_pressed,
     shift_is_pressed,
     space_is_pressed,
     show_hud;

u8 up_key,
   down_key,
   left_key, turn_left_key,
   right_key, turn_right_key,
   forward_key,
   backward_key,
   toggle_HUD_key,
   toggle_BVH_key,
   toggle_SSB_key,
   alt_key,
   ctrl_key,
   shift_key,
   space_key,
   exit_key;