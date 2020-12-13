#pragma once

#include "lib/core/perf.h"
#include "lib/core/str.h"
#include "lib/globals/display.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/scene.h"

void setControllerModeInHUD(bool fps) {
    char* mode = hud.mode;
    *mode++ = fps ? 'F' : 'O';
    *mode++ = fps ? 'p' : 'r';
    *mode   = fps ? 's' : 'b';
}

//void setAltModeInHUD(bool alt) {
//    char* alt_mode = hud.aux_alt;
//    *alt_mode++ = 'O';
//    *alt_mode++ = alt ? 'n' : 'f';
//    *alt_mode   = alt ? ' ' : 'f';
//}

void setRunOnInHUD(bool gpu) {
    char* run_on = hud.run_on;
    *run_on++ = gpu ? 'G' : 'C';
    *run_on++ = 'P';
    *run_on   = 'U';
}

void setUseBVH(bool yes) {
    char* use_bvh = hud.use_bvh;
    *use_bvh++ = yes ? 'Y' : ' ';
    *use_bvh++ = yes ? 'e' : 'N';
    *use_bvh   = yes ? 's' : 'o';
}

void setUseOld(bool old) {
    char* use_old = hud.use_old;
    *use_old++ = old ? 'O' : 'N';
    *use_old++ = old ? 'l' : 'e';
    *use_old   = old ? 'd' : 'w';
}

void initHUD() {
    hud.is_visible = true;

    char* str_template = "Width  : ___1\n"
                         "Height : ___2\n"
                         "Run on :  3__\n"
                         "Use BVH:  4__\n"
//                         "SSB    :  4__\n"
                         "FPS    : ___5\n"
                         "mic-s/f: ___6\n"
                         "Spheres: ___7\n"
                         "Pixels : __8%\n"
                         "Mode   :  9__\n";
//                         "* Alt. :  8__\n"
//                         "* Ms/F : ___9\n";

    char* HUD_char = str_template;
    char* HUD_text_char = hud.text;

    while (*HUD_char) {
        switch (*HUD_char) {
            case '1':  hud.width = HUD_text_char; break;
            case '2':  hud.height = HUD_text_char; break;
            case '3':  hud.run_on = HUD_text_char; break;
            case '4':  hud.use_bvh = HUD_text_char; break;
//            case '4':  hud.use_old = HUD_text_char; break;
            case '5':  hud.fps = HUD_text_char; break;
            case '6':  hud.msf = HUD_text_char; break;
            case '7':  hud.spr = HUD_text_char; break;
            case '8':  hud.pixels = HUD_text_char; break;
            case '9':  hud.mode = HUD_text_char; break;
//            case '8':  hud.aux_alt = HUD_text_char; break;
//            case '9':  hud.aux_msf = HUD_text_char; break;
        }

        *HUD_text_char++ = *HUD_char++;
    }
    *HUD_text_char = '\0';

    setControllerModeInHUD(false);
}

inline void updateHUDCounters(Timer *timer, u8 visible_nodes, u32 active_pixels) {
    printNumberIntoString(timer->average_frames_per_second, hud.fps);
    printNumberIntoString(timer->average_microseconds_per_frame, hud.msf);
//    printNumberIntoString(aux->average_milliseconds_per_frame, hud.aux_msf);
    printNumberIntoString(visible_nodes, hud.spr);
    printNumberIntoString((u16)(100.0f * ((f32)active_pixels / (f32)frame_buffer.size)), hud.pixels);
}

inline void updateHUDDimensions() {
    printNumberIntoString(frame_buffer.width, hud.width);
    printNumberIntoString(frame_buffer.height, hud.height);
}