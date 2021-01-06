#pragma once

#include <math.h>

#include "lib/core/types.h"

#define MAX_WIDTH 3840
#define MAX_HEIGHT 2160

#define PIXEL_SIZE 4
#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)

#define HUD_LENGTH 140
#define HUD_WIDTH 12
#define HUD_RIGHT 100
#define HUD_TOP 10

typedef struct {
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

typedef struct {
    vec3* color;
    vec2i position;
    Color sliders;
    f32 factor;
    Bounds2Di R, G, B, RGB;
    bool is_visible,
         is_controlled,
         is_red_controlled,
         is_green_controlled,
         is_blue_controlled,
         is_rgb_controlled;
} ColorControl;
ColorControl color_control;

typedef struct {
    f32 *key_intensity, *fill_intensity, *rim_intensity;
    vec2i position;
    u8 key_slider, fill_slider, rim_slider;
    Bounds2Di key_bounds, fill_bounds, rim_bounds;
    bool is_visible,
         is_controlled,
         is_key_controlled,
         is_fill_controlled,
         is_rim_controlled;
} LightControls;
LightControls light_controlls;

typedef struct {
    vec2i position;
    vec3 *key_color, *fill_color, *rim_color, *ambient_color;
    Bounds2Di key_bounds, fill_bounds, rim_bounds, ambient_bounds;
    bool is_visible,
         is_key_selected,
         is_fill_selected,
         is_rim_selected,
         is_ambient_selected;
} LightSelector;
LightSelector light_selector;

typedef struct {
    u16 width, height;
    u32 width_times_height;
    f32 height_over_width,
        width_over_height,
        f_height, f_width,
        h_height, h_width;
} Dimentions;

#define HUD_COLOR 0x0000FF00

typedef union {
    Color color;
    u32 value;
} Pixel;

typedef struct FrameBuffer {
    Dimentions dimentions;
    Pixel* pixels;
} FrameBuffer;
FrameBuffer frame_buffer;

#ifdef __CUDACC__
    __device__ u32 d_pixels[MAX_WIDTH * MAX_HEIGHT];
    __constant__ Dimentions d_dimentions[1];
//    __constant__ u8 d_GAMMA_LUT[256];
#endif

void updateFrameBufferDimensions(u16 width, u16 height) {
    frame_buffer.dimentions.width = width;
    frame_buffer.dimentions.height = height;
    frame_buffer.dimentions.width_times_height = frame_buffer.dimentions.width * frame_buffer.dimentions.height;
    frame_buffer.dimentions.f_width  = (f32)frame_buffer.dimentions.width;
    frame_buffer.dimentions.f_height = (f32)frame_buffer.dimentions.height;
    frame_buffer.dimentions.h_width  = 0.5f * frame_buffer.dimentions.f_width;
    frame_buffer.dimentions.h_height = 0.5f * frame_buffer.dimentions.f_height;
    frame_buffer.dimentions.width_over_height = frame_buffer.dimentions.f_width / frame_buffer.dimentions.f_height;
    frame_buffer.dimentions.height_over_width = frame_buffer.dimentions.f_height / frame_buffer.dimentions.f_width;
#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_dimentions, &frame_buffer.dimentions, sizeof(Dimentions), 0, cudaMemcpyHostToDevice));
#endif
}

void initFrameBuffer() {
    frame_buffer.pixels = AllocN(Pixel, RENDER_SIZE);
    updateFrameBufferDimensions(MAX_WIDTH, MAX_HEIGHT);
}