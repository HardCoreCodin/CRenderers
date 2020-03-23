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

void initFrameBuffer(FrameBuffer* frame_buffer, Memory* memory) {
    frame_buffer->width = 3840;
    frame_buffer->height = 2160;
    frame_buffer->size = frame_buffer->width * frame_buffer->height;
    frame_buffer->pixels = (u32*)allocate(memory, RENDER_SIZE);
}