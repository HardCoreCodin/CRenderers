#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"

#define PIXEL_SIZE 4
#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)

FrameBuffer frame_buffer;
Color WHITE, BLACK;
void initFrameBuffer() {
    frame_buffer.width = MAX_WIDTH;
    frame_buffer.height = MAX_HEIGHT;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;
    frame_buffer.pixels = AllocN(Pixel, RENDER_SIZE);
    WHITE.R = 255;
    WHITE.G = 255;
    WHITE.B = 255;
    WHITE.A = 0;
    BLACK.R = 0;
    BLACK.G = 0;
    BLACK.B = 0;
    BLACK.A = 0;
}