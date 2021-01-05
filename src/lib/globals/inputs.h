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
      mouse_pos_raw_diff,
      mouse_movement;

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

typedef struct {
    u8 up,
       down,
       left, turn_left,
       right, turn_right,
       forward,
       backward,
       toggle_HUD,
       toggle_BVH,
       toggle_SSB,
       toggle_GPU,
       alt,
       ctrl,
       shift,
       space,
       exit,
       set_beauty,
       set_normal,
       set_depth,
       set_uvs;
} KeyMap;
KeyMap keys;