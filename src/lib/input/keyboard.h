#pragma once

#include "lib/core/types.h"
#include "lib/globals/app.h"
#include "lib/globals/inputs.h"

void keyChanged(u8 key, bool pressed) {
    if      (key == keys.turn_left) turn_left = pressed;
    else if (key == keys.turn_right) turn_right = pressed;
    else if (key == keys.left) move_left = pressed;
    else if (key == keys.right) move_right = pressed;
    else if (key == keys.up) move_up = pressed;
    else if (key == keys.down) move_down = pressed;
    else if (key == keys.forward) move_forward = pressed;
    else if (key == keys.backward) move_backward = pressed;

    else if (key == keys.ctrl) ctrl_is_pressed = pressed;
    else if (key == keys.alt) alt_is_pressed = pressed;
    else if (key == keys.shift) shift_is_pressed = pressed;
    else if (key == keys.space) space_is_pressed = pressed;

    else if (key == keys.exit) is_running = false;

    else if (key == keys.set_beauty && !pressed) render_mode = Beauty;
    else if (key == keys.set_normal && !pressed) render_mode = Normals;
    else if (key == keys.set_depth && !pressed) render_mode = Depth;
    else if (key == keys.set_uvs && !pressed) render_mode = UVs;

    else if (key == keys.toggle_HUD && !pressed) show_hud = !show_hud;
    else if (key == keys.toggle_BVH && !pressed) show_BVH = !show_BVH;
    else if (key == keys.toggle_SSB && !pressed) show_SSB = !show_SSB;
#ifdef __CUDACC__
    else if (key == keys.toggle_GPU && !pressed) use_GPU = !use_GPU;
#endif
}