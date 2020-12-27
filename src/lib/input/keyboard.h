#pragma once

#include "lib/core/types.h"
#include "lib/globals/app.h"
#include "lib/globals/inputs.h"

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
    else if (key == alt_key) alt_is_pressed = pressed;
    else if (key == shift_key) shift_is_pressed = pressed;
    else if (key == space_key) space_is_pressed = pressed;

    else if (key == exit_key) is_running = false;

    else if (key == set_beauty_key && !pressed) render_mode = Beauty;
    else if (key == set_normal_key && !pressed) render_mode = Normals;
    else if (key == set_depth_key && !pressed) render_mode = Depth;
    else if (key == set_uvs_key && !pressed) render_mode = UVs;

    else if (key == toggle_HUD_key && !pressed) show_hud = !show_hud;
    else if (key == toggle_BVH_key && !pressed) show_BVH = !show_BVH;
    else if (key == toggle_SSB_key && !pressed) show_SSB = !show_SSB;
#ifdef __CUDACC__
    else if (key == toggle_GPU_key && !pressed) use_GPU = !use_GPU;
#endif
}