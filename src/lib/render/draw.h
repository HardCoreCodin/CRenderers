#pragma once

#include "lib/core/types.h"
#include "lib/core/font.h"
#include "lib/render/buffers.h"


#define FIRST_CHARACTER_CODE 32
#define LAST_CHARACTER_CODE 127
#define LINE_HEIGHT 12

void drawCharacter(u8 character, u32 text_color, u32* pixel, u16 step) {
    u8* byte = Font + FONT_WIDTH*(character - FIRST_CHARACTER_CODE);

    for (int h = 0; h < FONT_HEIGHT ; h++) {
        for (int w = 0; w < FONT_WIDTH; w++) {
            /* skip background bits */
            if (*byte & (0x80 >> w))
                *pixel = text_color;

            pixel++;
        }
        byte++;
        pixel += step;
    }
}

void drawText(char *str, u32 text_color, int x, int y, u32* pixels, u16 width, u16 height) {
    if (x < 0 || x > width - FONT_WIDTH ||
        y < 0 || y > height - FONT_HEIGHT)
        return;

    u16 current_x = x;
    u16 current_y = y;

    u16 step = width - FONT_WIDTH;
    u32 line = width * LINE_HEIGHT;
    u32* pixel = pixels + width * y + x;;

    char character = *str;

    while (character) {
        if (character == '\n') {
            if (current_y + FONT_HEIGHT > height)
                break;

            pixel += line - current_x + x;
            current_x = x;
            current_y += LINE_HEIGHT;
        } else if (character == '\t')
            current_x += (4 - ((current_x / FONT_WIDTH) & 3)) * FONT_WIDTH;
        else if (character >= FIRST_CHARACTER_CODE &&
                 character <= LAST_CHARACTER_CODE)
            drawCharacter(character, text_color, pixel, step);

        character = *++str;
        current_x += FONT_WIDTH;
        pixel += FONT_WIDTH;
    }
}

//inline void drawRect(Pixel color, u16 width, u16 height, u32 starting_offset) {
//    pixel = frame_buffer.pixels + starting_offset;
//    u16 x, y;
//    for (y = 0; y < width; y++) {
//        for (x = 0; x < height; x++)
//            *(pixel + x) = color;
//
//        pixel += frame_buffer.width;
//    }
//}
//
//inline void drawLine(Pixel color, u32 x1, u32 y1, u32 x2, u32 y2) {
//
//}