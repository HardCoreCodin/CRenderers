#pragma once

#include "lib/core/types.h"

#define MAX_WIDTH 3840
#define MAX_HEIGHT 2160

#define PIXEL_SIZE 4
#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)

#define HUD_LENGTH 140
#define HUD_WIDTH 12
#define HUD_RIGHT 100
#define HUD_TOP 10

typedef struct HUD {
    bool is_visible;
    char text[HUD_LENGTH];
    char *width,
         *height,
         *run_on,
         *fps,
         *msf,
         *mode;
} HUD;
HUD hud;

#define HUD_COLOR 0x0000FF00
typedef struct FrameBuffer {
    u16 width, height;
    u32 size;
    f32 height_over_width,
        width_over_height,
        f_height, f_width,
        h_height, h_width;
    Pixel* pixels;
} FrameBuffer;
FrameBuffer frame_buffer;

void updateFrameBufferDimensions(u16 width, u16 height) {
    frame_buffer.width = width;
    frame_buffer.height = height;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;
    frame_buffer.f_width  = (f32)frame_buffer.width;
    frame_buffer.f_height = (f32)frame_buffer.height;
    frame_buffer.h_width  = 0.5f * frame_buffer.f_width;
    frame_buffer.h_height = 0.5f * frame_buffer.f_height;
    frame_buffer.width_over_height = frame_buffer.f_width / frame_buffer.f_height;
    frame_buffer.height_over_width = frame_buffer.f_height / frame_buffer.f_width;
}

void initFrameBuffer() {
    frame_buffer.pixels = AllocN(Pixel, RENDER_SIZE);
    updateFrameBufferDimensions(MAX_WIDTH, MAX_HEIGHT);
}

#ifdef __CUDACC__
    __device__ u32 d_pixels[MAX_WIDTH * MAX_HEIGHT];

    #define copyPixelsFromGPUtoCPU(pixels, count) \
        gpuErrchk(cudaMemcpyFromSymbol(pixels, d_pixels, sizeof(u32) * count, 0, cudaMemcpyDeviceToHost))
#endif