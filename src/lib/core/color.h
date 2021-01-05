#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/globals/display.h"
#include "lib/shapes/rectangle.h"

// Filmic Tone-Mapping: https://www.slideshare.net/naughty_dog/lighting-shading-by-john-hable
// ====================
// A = Shoulder Strength (i.e: 0.22)
// B = Linear Strength   (i.e: 0.3)
// C = Linear Angle      (i.e: 0.1)
// D = Toe Strength      (i.e: 0.2)
// E = Toe Numerator     (i.e: 0.01)
// F = Toe Denumerator   (i.e: 0.3)
// LinearWhite = Linear White Point Value (i.e: 11.2)
//   Note: E/F = Toe Angle
//   Note: i.e numbers are NOT gamma corrected(!)
//
// f(x) = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F
//
// FinalColor = f(LinearColor)/f(LinearWhite)
//
// i.e:
// x = max(0, LinearColor-0.004)
// GammaColor = (x*(6.2*x + 0.5))/(x*(6.2*x+1.7) + 0.06)
//
// A = 6.2
// B = 1.7
//
// C*B = 1/2
// D*F = 0.06
// D*E = 0
//
// C = 1/2*1/B = 1/2*1/1.7 = 1/(2*1.7) = 1/3.4 =
// D = 1
// F = 0.06
// E = 0

#define TONE_MAP__SHOULDER_STRENGTH 6.2f
#define TONE_MAP__TOE_STRENGTH 1
#define TONE_MAP__TOE_NUMERATOR 0
#define TONE_MAP__TOE_DENOMINATOR 1
#define TONE_MAP__TOE_ANGLE (TONE_MAP__TOE_NUMERATOR / TONE_MAP__TOE_DENOMINATOR)
#define TONE_MAP__LINEAR_ANGLE (1.0f/3.4f)
#define TONE_MAP__LINEAR_WHITE 1
#define TONE_MAP__LINEAR_STRENGTH 1
// LinearWhite = 1:
// f(LinearWhite) = f(1)
// f(LinearWhite) = (A + C*B + D*E)/(A + B + D*F) - E/F
#define TONE_MAPPED__LINEAR_WHITE ( \
    (                               \
        TONE_MAP__SHOULDER_STRENGTH + \
        TONE_MAP__LINEAR_ANGLE * TONE_MAP__LINEAR_STRENGTH + \
        TONE_MAP__TOE_STRENGTH * TONE_MAP__TOE_NUMERATOR \
    ) / (                           \
        TONE_MAP__SHOULDER_STRENGTH + TONE_MAP__LINEAR_STRENGTH + \
        TONE_MAP__TOE_STRENGTH * TONE_MAP__TOE_DENOMINATOR  \
    ) - TONE_MAP__TOE_ANGLE \
)

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 toneMapped(f32 LinearColor) {
    f32 x = LinearColor - 0.004f;
    if (x < 0.0f) x = 0.0f;
    f32 x2 = x*x;
    f32 x2_times_sholder_strength = x2 * TONE_MAP__SHOULDER_STRENGTH;
    f32 x_times_linear_strength   =  x * TONE_MAP__LINEAR_STRENGTH;
    return (
                   (
                           (
                                   x2_times_sholder_strength + x*x_times_linear_strength + TONE_MAP__TOE_STRENGTH*TONE_MAP__TOE_NUMERATOR
                           ) / (
                                   x2_times_sholder_strength +   x_times_linear_strength + TONE_MAP__TOE_STRENGTH*TONE_MAP__TOE_DENOMINATOR
                           )
                   ) - TONE_MAP__TOE_ANGLE
           ) / (TONE_MAPPED__LINEAR_WHITE);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 toneMappedBaked(f32 LinearColor) {
    // x = max(0, LinearColor-0.004)
    // GammaColor = (x*(6.2*x + 0.5))/(x*(6.2*x+1.7) + 0.06)
    // GammaColor = (x*x*6.2 + x*0.5)/(x*x*6.2 + x*1.7 + 0.06)

    f32 x = LinearColor - 0.004f;
    if (x < 0.0f) x = 0.0f;
    f32 x2_times_sholder_strength = x * x * 6.2f;
    return (x2_times_sholder_strength + x*0.5f)/(x2_times_sholder_strength + x*1.7f + 0.06f);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 gammaCorrected(f32 x) {
    return (x <= 0.0031308f ? (x * 12.92f) : (1.055f * powf(x, 1.0f/2.4f) - 0.055f));
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 gammaCorrectedApproximately(f32 x) {
    return powf(x, 1.0f/2.2f);
}


#define setPixelColor(pixel, color) \
        color.x *= 255; \
        color.y *= 255; \
        color.z *= 255; \
        pixel->color.R = color.x > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.x; \
        pixel->color.G = color.y > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.y; \
        pixel->color.B = color.z > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.z

#define setPixelToneMappedColor(pixel, color) \
        color.x = toneMapped(color.x);    \
        color.y = toneMapped(color.y);    \
        color.z = toneMapped(color.z);    \
        setPixelColor(pixel, color)

#define setPixelBakedToneMappedColor(pixel, color) \
        color.x = toneMappedBaked(color.x);    \
        color.y = toneMappedBaked(color.y);    \
        color.z = toneMappedBaked(color.z);    \
        setPixelColor(pixel, color)

#define setPixelGammaCorrectedColor(pixel, color) \
        color.x = gammaCorrected(color.x);    \
        color.y = gammaCorrected(color.y);    \
        color.z = gammaCorrected(color.z);    \
        setPixelColor(pixel, color)

#define setPixelApproximatedGammaCorrectedColor(pixel, color) \
        color.x = gammaCorrectedApproximately(color.x);    \
        color.y = gammaCorrectedApproximately(color.y);    \
        color.z = gammaCorrectedApproximately(color.z);    \
        setPixelColor(pixel, color)

#define COLOR_CONTROL__POSITION_X 40
#define COLOR_CONTROL__POSITION_Y 40
#define COLOR_CONTROL__SLIDER_WIDTH 4
#define COLOR_CONTROL__SLIDER_RANGE 128
#define COLOR_CONTROL__SLIDER_HEIGHT (COLOR_CONTROL__SLIDER_RANGE)
#define COLOR_CONTROL__MARKER_WIDTH 40
#define COLOR_CONTROL__MARKER_HEIGHT 20

void initColorControl() {
    color_control.is_visible = false;
    color_control.is_controlled = false;
    color_control.is_red_controlled = false;
    color_control.is_blue_controlled = false;
    color_control.is_green_controlled = false;
    color_control.position.x = COLOR_CONTROL__POSITION_X;
    color_control.position.y = COLOR_CONTROL__POSITION_Y;
}

inline void setColorControlRedBounds() {
    color_control.R.x_range.min = color_control.position.x;
    color_control.R.y_range.min = color_control.position.y + COLOR_CONTROL__SLIDER_HEIGHT - color_control.sliders.R;
    color_control.R.x_range.max = color_control.R.x_range.min + COLOR_CONTROL__MARKER_WIDTH;
    color_control.R.y_range.max = color_control.R.y_range.min + COLOR_CONTROL__MARKER_HEIGHT;
}

inline void setColorControlGreenBounds() {
    color_control.G.x_range.min = color_control.position.x + COLOR_CONTROL__MARKER_WIDTH;
    color_control.G.y_range.min = color_control.position.y + COLOR_CONTROL__SLIDER_HEIGHT - color_control.sliders.G;
    color_control.G.x_range.max = color_control.G.x_range.min + COLOR_CONTROL__MARKER_WIDTH;
    color_control.G.y_range.max = color_control.G.y_range.min + COLOR_CONTROL__MARKER_HEIGHT;
}

inline void setColorControlBlueBounds() {
    color_control.B.x_range.min = color_control.position.x + COLOR_CONTROL__MARKER_WIDTH + COLOR_CONTROL__MARKER_WIDTH;
    color_control.B.y_range.min = color_control.position.y + COLOR_CONTROL__SLIDER_HEIGHT - color_control.sliders.B;
    color_control.B.x_range.max = color_control.B.x_range.min + COLOR_CONTROL__MARKER_WIDTH;
    color_control.B.y_range.max = color_control.B.y_range.min + COLOR_CONTROL__MARKER_HEIGHT;
}

inline void bindColorControl(vec3* color) {
    color_control.color = color;
    color_control.sliders.R = (u8)(COLOR_CONTROL__SLIDER_RANGE * color->x);
    color_control.sliders.G = (u8)(COLOR_CONTROL__SLIDER_RANGE * color->y);
    color_control.sliders.B = (u8)(COLOR_CONTROL__SLIDER_RANGE * color->z);
    setColorControlRedBounds();
    setColorControlGreenBounds();
    setColorControlBlueBounds();
}

inline void updateRedColorControl(vec2i movement) {
    i32 new_slider_pos = (i32)color_control.sliders.R - (i32)movement.y;
    if (new_slider_pos <= 0) {
        color_control.color->x = 0;
        color_control.sliders.R = 0;
    } else if (new_slider_pos >= COLOR_CONTROL__SLIDER_RANGE) {
        color_control.color->x = 1;
        color_control.sliders.R = COLOR_CONTROL__SLIDER_RANGE;
    } else {
        color_control.color->x -= ((f32)movement.y / (f32)COLOR_CONTROL__SLIDER_RANGE);
        color_control.sliders.R -= movement.y;
    }

    setColorControlRedBounds();
}

inline void updateGreenColorControl(vec2i movement) {
    i32 new_slider_pos = (i32)color_control.sliders.G - (i32)movement.y;
    if (new_slider_pos <= 0) {
        color_control.color->y = 0;
        color_control.sliders.G = 0;
    } else if (new_slider_pos >= COLOR_CONTROL__SLIDER_RANGE) {
        color_control.color->y = 1;
        color_control.sliders.G = COLOR_CONTROL__SLIDER_RANGE;
    } else {
        color_control.color->y -= ((f32)movement.y / (f32)COLOR_CONTROL__SLIDER_RANGE);
        color_control.sliders.G -= movement.y;
    }

    setColorControlGreenBounds();
}

inline void updateBlueColorControl(vec2i movement) {
    i32 new_slider_pos = (i32)color_control.sliders.B - (i32)movement.y;
    if (new_slider_pos <= 0) {
        color_control.color->z = 0;
        color_control.sliders.B = 0;
    } else if (new_slider_pos >= COLOR_CONTROL__SLIDER_RANGE) {
        color_control.color->z = 1;
        color_control.sliders.B = COLOR_CONTROL__SLIDER_RANGE;
    } else {
        color_control.color->z -= ((f32)movement.y / (f32)COLOR_CONTROL__SLIDER_RANGE);
        color_control.sliders.B -= movement.y;
    }

    setColorControlBlueBounds();
}

void drawColorControl() {
    Pixel R, G, B, border;
    border.color = WHITE;
    R.color = RED;
    G.color = GREEN;
    B.color = BLUE;

    u32 x = color_control.position.x;
    u32 y = color_control.position.y;
    u32 half_slider_width = COLOR_CONTROL__SLIDER_WIDTH / 2;
    u32 half_marker_width = COLOR_CONTROL__MARKER_WIDTH / 2;
    u32 half_marker_height = COLOR_CONTROL__MARKER_HEIGHT / 2;
    u32 offset = half_marker_width - half_slider_width;

    // Draw sliders:
    Bounds2Di rect;
    rect.y_range.min = y + half_marker_height;
    rect.y_range.max = rect.y_range.min + COLOR_CONTROL__SLIDER_HEIGHT;
    rect.x_range.min = x + offset;
    rect.x_range.max = rect.x_range.min + COLOR_CONTROL__SLIDER_WIDTH;
    fillRect(&rect, R);

    rect.x_range.min += COLOR_CONTROL__MARKER_WIDTH;
    rect.x_range.max += COLOR_CONTROL__MARKER_WIDTH;
    fillRect(&rect, G);

    rect.x_range.min += COLOR_CONTROL__MARKER_WIDTH;
    rect.x_range.max += COLOR_CONTROL__MARKER_WIDTH;
    fillRect(&rect, B);

    R.color.R = (u8)(255.0f * color_control.color->x);
    G.color.G = (u8)(255.0f * color_control.color->y);
    B.color.B = (u8)(255.0f * color_control.color->z);

    fillRect(&color_control.R, R);
    fillRect(&color_control.G, G);
    fillRect(&color_control.B, B);
    drawRect(&color_control.R, border);
    drawRect(&color_control.G, border);
    drawRect(&color_control.B, border);

    Pixel color_pixel;
    vec3 color = *color_control.color;
    Bounds2Di color_rect;
    color_rect.x_range.min = color_control.R.x_range.min;
    color_rect.x_range.max = color_control.B.x_range.max;
    color_rect.y_range.min = rect.y_range.max + COLOR_CONTROL__MARKER_HEIGHT;
    color_rect.y_range.max = rect.y_range.max + COLOR_CONTROL__MARKER_HEIGHT*2;
    setPixelColor((&color_pixel), color);
    fillRect(&color_rect, color_pixel);
    drawRect(&color_rect, border);
}

//u8 GAMMA_LUT[256];
//
//#define setPixelColor1(pixel, color, gamma_lut) \
//        color.x *= 255.0f; \
//        color.y *= 255.0f; \
//        color.z *= 255.0f; \
//        pixel->color.R = color.x > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : gamma_lut[(u8)color.x]; \
//        pixel->color.G = color.y > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : gamma_lut[(u8)color.y]; \
//        pixel->color.B = color.z > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : gamma_lut[(u8)color.z]
//
//void initGammaLUT() {
//    f32 value;
//    for (u16 i = 0; i < 256; i++) {
//        value = (f32)i;
//        value /= 256;
//        value = value <= 0.0031308f ? (value * 12.92f) : (1.055f * powf(value, 1.0f/2.4f) - 0.055f);
////        value = powf(value, 1.0f/2.2f);
//        value *= 256;
//        GAMMA_LUT[i] = (u8)value;
//    }
//
//#ifdef __CUDACC__
//    gpuErrchk(cudaMemcpyToSymbol(d_GAMMA_LUT, GAMMA_LUT, 256, 0, cudaMemcpyHostToDevice));
//#endif
//}