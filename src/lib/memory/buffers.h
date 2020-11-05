#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"

#define PIXEL_SIZE 4
#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)

FrameBuffer frame_buffer;
Color WHITE;
void initFrameBuffer() {
    frame_buffer.width = 3840;
    frame_buffer.height = 2160;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;
    frame_buffer.pixels = AllocN(Pixel, RENDER_SIZE);
    WHITE.R = 255;
    WHITE.G = 255;
    WHITE.B = 255;
    WHITE.A = 0;
}