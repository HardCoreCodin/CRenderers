#pragma once

#include "lib/core/types.h"
#include "lib/core/font.h"
#include "lib/render/buffers.h"


#define FIRST_CHARACTER_CODE 32
#define LAST_CHARACTER_CODE 127
#define LINE_HEIGHT 12

static u32 character_buffer[4 * FONT_SIZE];

int drawCharacter(u8 character, u32 text_color, u32 pixel_offset, FrameBuffer* frame_buffer) {
    if (character < FIRST_CHARACTER_CODE ||
        character > LAST_CHARACTER_CODE)
        return 0;

    u8* byte = Font + (FONT_WIDTH * (character - FIRST_CHARACTER_CODE));
    u32* character_buffer_pixel = character_buffer;

    int i, j = 0;
    for (i = 0; i < FONT_SIZE; i++) {
        *character_buffer_pixel++ = (*byte & (0x80 >> j++)) ? text_color : 0;
        if (j > 7) {
            j = 0;
            byte++;
        }
    }

    u16 step = frame_buffer->width - FONT_WIDTH;
    character_buffer_pixel = character_buffer;
    u32* pixel = frame_buffer->pixels + pixel_offset;

    for (j = FONT_HEIGHT; j >= 0 ; j--) {
        for (i = 0; i < FONT_WIDTH; i++) {
            int idx = j * FONT_WIDTH + i;

            /* skip background bits */
            if (character_buffer[idx])
                *pixel = character_buffer[idx];

            pixel++;
        }
        pixel += step;
    }
    return 1;
}

int drawString(const char *str, u32 text_color, int x, int y, FrameBuffer* frame_buffer) {
    u16 w = frame_buffer->width;
    u16 h = frame_buffer->height;

    if (x < 0 ||
        y < 0 ||
        w < x + FONT_WIDTH ||
        h < y + FONT_HEIGHT)
        return 0;

    int current_x = x;
    int current_y = y;

    u32 line_offset  = w * LINE_HEIGHT;
    u32 pixel_offset = w * y + x;

    while (*str) {
        if (*str == '\n') {
            pixel_offset -= line_offset + current_x - x;

            current_x = x;
            current_y -= LINE_HEIGHT;

            if (!*++str)
                break;
        }

        if (current_y < 0 || w < current_y + FONT_HEIGHT)
            return 0;

        if (*str == '\t') {
            current_x += (4 - ((current_x / FONT_WIDTH) & 3)) * FONT_WIDTH;
            str++;
            continue;
        }

        drawCharacter(*str++, text_color, pixel_offset, frame_buffer);
        current_x += FONT_WIDTH;
        pixel_offset += FONT_WIDTH;
    }

    return 0;
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