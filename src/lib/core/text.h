#pragma once

#include "lib/core/types.h"
#include "lib/core/font.h"
#include "lib/memory/buffers.h"

#define LINE_HEIGHT 12
#define FIRST_CHARACTER_CODE 32
#define LAST_CHARACTER_CODE 127

void drawText(char *str, u32 color, int x, int y) {
    if (x < 0 || x > frame_buffer.width - FONT_WIDTH ||
        y < 0 || y > frame_buffer.height - FONT_HEIGHT)
        return;

    u16 current_x = x;
    u16 current_y = y;
    u16 t_offset = 0;
    u16 pixel_line_step = frame_buffer.width - FONT_WIDTH;
    u32 char_line_step  = frame_buffer.width * LINE_HEIGHT;
    u32* pixel = frame_buffer.pixels + frame_buffer.width * y + x;;
    u32* character_pixel;
    u8* byte;
    char character = *str;

    while (character) {
        if (character == '\n') {
            if (current_y + FONT_HEIGHT > frame_buffer.height)
                break;

            pixel += char_line_step - current_x + x;
            current_x = x;
            current_y += LINE_HEIGHT;
        } else if (character == '\t') {
            t_offset = FONT_WIDTH * (4 - ((current_x / FONT_WIDTH) & 3));
            current_x += t_offset;
            pixel += t_offset;
        } else if (character >= FIRST_CHARACTER_CODE &&
                   character <= LAST_CHARACTER_CODE) {
            byte = Font + FONT_WIDTH*(character - FIRST_CHARACTER_CODE);
            character_pixel = pixel;
            for (int h = 0; h < FONT_HEIGHT ; h++) {
                for (int w = 0; w < FONT_WIDTH; w++) {
                    /* skip background bits */
                    if (*byte & (0x80 >> w))
                        *character_pixel = color;

                    character_pixel++;
                }
                byte++;
                character_pixel += pixel_line_step;
            }

            current_x += FONT_WIDTH;
            pixel += FONT_WIDTH;
        }
        character = *++str;
    }
}