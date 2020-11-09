#pragma once

#include "lib/core/perf.h"
#include "lib/core/string.h"
#include "lib/input/keyboard.h"
#include "lib/memory/buffers.h"
#include "lib/nodes/scene.h"

#define HUD_WIDTH 12
#define HUD_RIGHT 100
#define HUD_TOP 10
#define HUD_COLOR 0x0000FF00

HUD hud;

void setControllerModeInHUD(bool fps) {
    char* mode = hud.mode;
    *mode++ = fps ? 'F' : 'O';
    *mode++ = fps ? 'p' : 'r';
    *mode   = fps ? 's' : 'b';
}

void initHUD() {
    hud.is_visible = true;

    char* template = "Width  : ___1\n"
                     "Height : ___2\n"
                     "FPS    : ___3\n"
                     "Ms/F   : ___4\n"
                     "Spheres: ___5\n"
                     "Pixels : __6%\n"
                     "Mode   :  7__\n";

    char* HUD_char = template;
    char* HUD_text_char = hud.text;

    while (*HUD_char) {
        switch (*HUD_char) {
            case '1':  hud.width = HUD_text_char; break;
            case '2':  hud.height = HUD_text_char; break;
            case '3':  hud.fps = HUD_text_char; break;
            case '4':  hud.msf = HUD_text_char; break;
            case '5':  hud.spr = HUD_text_char; break;
            case '6':  hud.pixels = HUD_text_char; break;
            case '7':  hud.mode = HUD_text_char; break;
        }

        *HUD_text_char++ = *HUD_char++;
    }
    *HUD_text_char = '\0';

    setControllerModeInHUD(false);
}

inline void updateHUDCounters(Timer *timer) {
    printNumberIntoString(timer->average_frames_per_second, hud.fps);
    printNumberIntoString(timer->average_milliseconds_per_frame, hud.msf);
    printNumberIntoString(scene.active_sphere_count, hud.spr);
    printNumberIntoString((u16)(100.0f * ((f32)frame_buffer.active_pixel_count / (f32)frame_buffer.size)), hud.pixels);
}

inline void updateHUDDimensions() {
    printNumberIntoString(frame_buffer.width, hud.width);
    printNumberIntoString(frame_buffer.height, hud.height);
}