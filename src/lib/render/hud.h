#pragma once

#include "lib/core/memory.h"
#include "lib/controls/controls.h"

#define HUD_LENGTH 48
#define HUD_LEFT 10
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
} HUD;

static HUD hud;

void setControllerModeInHUD(bool fps, char* mode) {
    *mode++ = fps ? 'F' : 'O';
    *mode++ = fps ? 'p' : 'r';
    *mode   = fps ? 's' : 'b';
}

void initHUD() {
    hud.is_visible = true;

    char* template = "___1 x ___2\n___3 f/s\n___4 ms/f\n5__";
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
    *HUD_text_char = '\0';

    setControllerModeInHUD(false, hud.mode);
}