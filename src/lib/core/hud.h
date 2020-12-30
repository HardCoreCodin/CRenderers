#pragma once

#include "lib/core/perf.h"
#include "lib/core/str.h"
#include "lib/globals/app.h"
#include "lib/globals/display.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/scene.h"

inline void setCountersInHUD(Timer *timer) {
    printNumberIntoString(timer->average_frames_per_second, hud.fps);
    printNumberIntoString(timer->average_microseconds_per_frame, hud.msf);
}

inline void setDimesionsInHUD() {
    printNumberIntoString(frame_buffer.dimentions.width, hud.width);
    printNumberIntoString(frame_buffer.dimentions.height, hud.height);
}

inline void setRenderModeInHUD() {
    char* mode = hud.mode;
    switch (render_mode) {
        case Normals: *mode++ = 'N'; *mode++ = 'o'; *mode++ = 'r'; *mode++ = 'm'; *mode++ = 'a'; *mode = 'l'; break;
        case Beauty : *mode++ = 'B'; *mode++ = 'e'; *mode++ = 'a'; *mode++ = 'u'; *mode++ = 't'; *mode = 'y'; break;
        case Depth  : *mode++ = ' '; *mode++ = 'D'; *mode++ = 'e'; *mode++ = 'p'; *mode++ = 't'; *mode = 'h'; break;
        case UVs    : *mode++ = 'T'; *mode++ = 'e'; *mode++ = 'x'; *mode++ = 'C'; *mode++ = 'o'; *mode = 'r'; break;
    }
}

inline void setRunOnInHUD() {
    char* run_on = hud.run_on;
    *run_on++ = use_GPU ? 'G' : 'C';
    *run_on++ = 'P';
    *run_on   = 'U';
}

void initHUD() {
    hud.is_visible = true;

    char* str_template = "Width  : ___1\n"
                         "Height : ___2\n"
                         "Using  :  3__\n"
                         "FPS    : ___4\n"
                         "mic-s/f: ___5\n"
                         "Mode : 6_____\n";

    char* HUD_char = str_template;
    char* HUD_text_char = hud.text;

    while (*HUD_char) {
        switch (*HUD_char) {
            case '1':  hud.width  = HUD_text_char; break;
            case '2':  hud.height = HUD_text_char; break;
            case '3':  hud.run_on = HUD_text_char; break;
            case '4':  hud.fps    = HUD_text_char; break;
            case '5':  hud.msf    = HUD_text_char; break;
            case '6':  hud.mode   = HUD_text_char; break;
        }

        *HUD_text_char++ = *HUD_char++;
    }
    *HUD_text_char = '\0';

    setCountersInHUD(&update_timer);
    setDimesionsInHUD();
    setRenderModeInHUD();
    setRunOnInHUD();
}
