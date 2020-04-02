#pragma once

#include "lib/core/perf.h"
#include "lib/core/string.h"
#include "lib/core/inputs.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"

#define HUD_LENGTH 100
#define HUD_WIDTH 12
#define HUD_RIGHT 100
#define HUD_TOP 10
#define HUD_COLOR 0x0000FF00

typedef struct HUD {
    bool is_visible;
    char text[HUD_LENGTH];
    char* width;
    char* height;
    char* fps;
    char* msf;
    char* mode;
    char* perf;

    Perf main_perf, debug_perf;
} HUD;

static HUD hud;

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
                     "Mode   :  5__\n"
                     "Perf   :     ";

    char* HUD_char = template;
    char* HUD_text_char = hud.text;

    while (*HUD_char) {
        switch (*HUD_char) {
            case '1':  hud.width = HUD_text_char; break;
            case '2':  hud.height = HUD_text_char; break;
            case '3':  hud.fps = HUD_text_char; break;
            case '4':  hud.msf = HUD_text_char; break;
            case '5':  hud.mode = HUD_text_char; break;
        }

        *HUD_text_char++ = *HUD_char++;
    }
    hud.perf = HUD_text_char - 1;
    *HUD_text_char = '\0';

    setControllerModeInHUD(false);

    initPerf(&hud.main_perf);
    initPerf(&hud.debug_perf);
}

inline void updateHUDCounters() {
    printNumberIntoString(hud.main_perf.avg.frames_per_second, hud.fps);
    printNumberIntoString(hud.main_perf.avg.milliseconds_per_frame, hud.msf);
}

inline void updateHUDDimensions() {
    printNumberIntoString(frame_buffer.width, hud.width);
    printNumberIntoString(frame_buffer.height, hud.height);
}