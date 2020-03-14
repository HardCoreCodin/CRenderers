#pragma once
#include "core.h"

inline void drawRect(Pixel color, u16 width, u16 height, u32 starting_offset) {
    pixel = frame_buffer.pixels + starting_offset;
    u16 x, y;
    for (y = 0; y < width; y++) {
        for (x = 0; x < height; x++)
            *(pixel + x) = color;

        pixel += frame_buffer.width;
    }
}

inline void drawLine(Pixel color, u32 x1, u32 y1, u32 x2, u32 y2) {
    
}