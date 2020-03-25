#pragma once

#include "lib/core/types.h"
#include "lib/core/memory.h"

#define PIXEL_SIZE 4
#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)

typedef struct FrameBuffer {
    u16 width, height;
    u32 size;
    u32* pixels;
} FrameBuffer;

static FrameBuffer frame_buffer = {3840, 2160};

void initFrameBuffer() {
    frame_buffer.size = frame_buffer.width * frame_buffer.height;
    frame_buffer.pixels = (u32*)allocate(RENDER_SIZE);
}