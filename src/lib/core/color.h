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

#define CONTROL__SLIDER_RANGE 128
#define CONTROL__SLIDER_LENGTH (CONTROL__SLIDER_RANGE)
#define CONTROL__LENGTH 20
#define CONTROL__THICKNESS 10

#define COLOR_CONTROL__RED_MARKER_WIDTH (CONTROL__LENGTH)
#define COLOR_CONTROL__RED_MARKER_HEIGHT (CONTROL__THICKNESS)
#define COLOR_CONTROL__GREEN_MARKER_WIDTH (CONTROL__LENGTH)
#define COLOR_CONTROL__GREEN_MARKER_HEIGHT (CONTROL__LENGTH)
#define COLOR_CONTROL__BLUE_MARKER_WIDTH (CONTROL__THICKNESS)
#define COLOR_CONTROL__BLUE_MARKER_HEIGHT (CONTROL__LENGTH)

inline void setColorControlRedBounds() {
    color_control.R.x_range.min = color_control.position.x + color_control.sliders.R - COLOR_CONTROL__RED_MARKER_WIDTH/2;
    color_control.R.y_range.min = color_control.position.y + CONTROL__SLIDER_LENGTH - COLOR_CONTROL__RED_MARKER_HEIGHT / 2;
    color_control.R.x_range.max = color_control.R.x_range.min + COLOR_CONTROL__RED_MARKER_WIDTH;
    color_control.R.y_range.max = color_control.R.y_range.min + COLOR_CONTROL__RED_MARKER_HEIGHT;
}

inline void setColorControlGreenBounds() {
    color_control.G.x_range.min = color_control.position.x - color_control.sliders.G + COLOR_CONTROL__GREEN_MARKER_WIDTH/2 + CONTROL__SLIDER_LENGTH * 2;
    color_control.G.y_range.min = color_control.position.y + color_control.sliders.G - COLOR_CONTROL__GREEN_MARKER_HEIGHT/2 - COLOR_CONTROL__RED_MARKER_HEIGHT/2;
    color_control.G.x_range.max = color_control.G.x_range.min + COLOR_CONTROL__GREEN_MARKER_WIDTH;
    color_control.G.y_range.max = color_control.G.y_range.min + COLOR_CONTROL__GREEN_MARKER_HEIGHT;
}

inline void setColorControlBlueBounds() {
    color_control.B.x_range.min = color_control.position.x + COLOR_CONTROL__RED_MARKER_WIDTH/2 + CONTROL__SLIDER_LENGTH;
    color_control.B.y_range.min = color_control.position.y + COLOR_CONTROL__RED_MARKER_HEIGHT/2 + CONTROL__SLIDER_LENGTH * 2 - color_control.sliders.B;
    color_control.B.x_range.max = color_control.B.x_range.min + COLOR_CONTROL__BLUE_MARKER_WIDTH;
    color_control.B.y_range.max = color_control.B.y_range.min + COLOR_CONTROL__BLUE_MARKER_HEIGHT;
}

inline void setColorControlRGBBounds() {
    color_control.RGB.x_range.min = color_control.R.x_range.min;
    color_control.RGB.y_range.min = color_control.B.y_range.min;
    color_control.RGB.x_range.max = color_control.R.x_range.max;
    color_control.RGB.y_range.max = color_control.B.y_range.max;
}

inline void bindColorControl(vec3* color, f32 factor) {
    color_control.factor = factor;
    color_control.color = color;
    color_control.sliders.R = (u8)(CONTROL__SLIDER_RANGE * color->x * 1.0f / factor);
    color_control.sliders.G = (u8)(CONTROL__SLIDER_RANGE * color->y * 1.0f / factor);
    color_control.sliders.B = (u8)(CONTROL__SLIDER_RANGE * color->z * 1.0f / factor);
    setColorControlRedBounds();
    setColorControlGreenBounds();
    setColorControlBlueBounds();
    setColorControlRGBBounds();
}

inline void updateColorControlComponent(u8 *slider, f32 *value, i32 diff, f32 factor) {
    i32 new_slider_pos = (i32)(*slider) + diff;
    if (new_slider_pos <= 0) {
        *value  = 0;
        *slider = 0;
    } else if (new_slider_pos >= CONTROL__SLIDER_RANGE) {
        *value  = factor;
        *slider = CONTROL__SLIDER_RANGE;
    } else {
        *value  += ((f32)diff / (f32)CONTROL__SLIDER_RANGE) * factor;
        *slider += diff;
    }
}


inline void updateRedColorControl(vec2i movement) {
    updateColorControlComponent(&color_control.sliders.R, &color_control.color->x, movement.x, color_control.factor);
    setColorControlRedBounds();
    setColorControlRGBBounds();
}

inline void updateGreenColorControl(vec2i movement) {
    updateColorControlComponent(&color_control.sliders.G, &color_control.color->y, abs(movement.y) > abs(movement.x) ? movement.y : -movement.x, color_control.factor);
    setColorControlGreenBounds();
}

inline void updateBlueColorControl(vec2i movement) {
    updateColorControlComponent(&color_control.sliders.B, &color_control.color->z, -movement.y, color_control.factor);
    setColorControlBlueBounds();
    setColorControlRGBBounds();
}

void setColorControlPosition(u16 position_x, u16 position_y) {
    color_control.position.x = position_x;
    color_control.position.y = position_y;
    setColorControlRedBounds();
    setColorControlGreenBounds();
    setColorControlBlueBounds();
    setColorControlRGBBounds();
}

void initColorControl(u16 initial_position_x, u16 initial_position_y) {
    color_control.is_visible = false;
    color_control.is_controlled = false;
    color_control.is_rgb_controlled = false;
    color_control.is_red_controlled = false;
    color_control.is_blue_controlled = false;
    color_control.is_green_controlled = false;
    setColorControlPosition(initial_position_x, initial_position_y);
}

void drawColorControl() {
    Pixel R, G, B, RGB, border, inactive_border;
    border.color = WHITE;
    inactive_border.color = GREY;
    R.color = RED;
    G.color = GREEN;
    B.color = BLUE;

    u32 x = color_control.position.x;
    u32 y = color_control.position.y;

    u32 red_start, red_end, red_at, blue_start, blue_end, blue_at, green_start_x, green_end_x, green_start_y, green_end_y;

    red_start = red_end = x;
    red_end  += CONTROL__SLIDER_LENGTH;
    red_at    = CONTROL__SLIDER_LENGTH + y;

    u32 blue_offset = COLOR_CONTROL__RED_MARKER_WIDTH/2 + COLOR_CONTROL__BLUE_MARKER_WIDTH/2;
    blue_start = blue_end = y + blue_offset + CONTROL__SLIDER_LENGTH;
    blue_end  += CONTROL__SLIDER_LENGTH;
    blue_at    = CONTROL__SLIDER_LENGTH + x + blue_offset;

    green_start_x = x + COLOR_CONTROL__GREEN_MARKER_WIDTH + CONTROL__SLIDER_LENGTH;
    green_start_y = y - COLOR_CONTROL__RED_MARKER_HEIGHT / 2 + CONTROL__SLIDER_LENGTH;
    green_end_x   = green_start_x + CONTROL__SLIDER_LENGTH;
    green_end_y   = green_start_y - CONTROL__SLIDER_LENGTH;

    Bounds2Di main_rect;
    main_rect.x_range.min = red_start;
    main_rect.x_range.max = red_end;
    main_rect.y_range.min = blue_start;
    main_rect.y_range.max = blue_end;

    vec2i rect_offset;
    rect_offset.x = green_start_x - red_end + CONTROL__SLIDER_LENGTH - color_control.sliders.G - COLOR_CONTROL__RED_MARKER_WIDTH;
    rect_offset.y = green_start_y - blue_start + color_control.sliders.G - CONTROL__SLIDER_LENGTH + COLOR_CONTROL__RED_MARKER_WIDTH;

    drawHLine2D(main_rect.x_range.min + rect_offset.x, main_rect.x_range.max + rect_offset.x, main_rect.y_range.min + rect_offset.y, G);
    drawVLine2D(main_rect.y_range.min + rect_offset.y, main_rect.y_range.max + rect_offset.y, main_rect.x_range.max + rect_offset.x, G);

    rect_offset.x = green_start_x - red_end + CONTROL__SLIDER_LENGTH - COLOR_CONTROL__RED_MARKER_WIDTH;
    rect_offset.y = green_start_y - blue_start - CONTROL__SLIDER_LENGTH + COLOR_CONTROL__RED_MARKER_WIDTH;

    drawHLine2D(main_rect.x_range.min + rect_offset.x, main_rect.x_range.max + rect_offset.x, main_rect.y_range.min + rect_offset.y, border);
    drawVLine2D(main_rect.y_range.min + rect_offset.y, main_rect.y_range.max + rect_offset.y, main_rect.x_range.max + rect_offset.x, border);

    drawLine2D(main_rect.x_range.min, main_rect.y_range.min, main_rect.x_range.min + rect_offset.x, main_rect.y_range.min + rect_offset.y, border);
    drawLine2D(main_rect.x_range.max, main_rect.y_range.max, main_rect.x_range.max + rect_offset.x, main_rect.y_range.max + rect_offset.y, border);
    drawLine2D(main_rect.x_range.max, main_rect.y_range.min, main_rect.x_range.max + rect_offset.x, main_rect.y_range.min + rect_offset.y, border);

    drawHLine2D(red_start, red_end, red_at, R);
    drawHLine2D(red_start, red_end, red_at+1, R);
    drawHLine2D(red_start, red_end, red_at-1, R);
    drawVLine2D(blue_start, blue_end, blue_at, B);
    drawVLine2D(blue_start, blue_end, blue_at+1, B);
    drawVLine2D(blue_start, blue_end, blue_at-1, B);
    drawLine2D(green_start_x, green_start_y, green_end_x, green_end_y, G);
    drawLine2D(green_start_x+1, green_start_y, green_end_x+1, green_end_y, G);
    drawLine2D(green_start_x, green_start_y-1, green_end_x, green_end_y-1, G);

    // Fill Color Picker Rectangle:
    RGB.color.G = (u8)(255.0f * gammaCorrected(color_control.color->y));

    vec3 color;
    color.z = 0;
    f32 step = 1.0f / (f32)CONTROL__SLIDER_RANGE;
    u32 pixel_index_start = frame_buffer.dimentions.width * main_rect.y_range.max;
    u32 pixel_index;
    for (y = 0; y < CONTROL__SLIDER_RANGE; y++, color.z += step, pixel_index_start -= frame_buffer.dimentions.width) {
        RGB.color.B = (u8)(255.0f * gammaCorrected(color.z));

        color.x = 0;
        pixel_index = pixel_index_start + main_rect.x_range.min;
        for (x = 0; x < CONTROL__SLIDER_RANGE; x++, pixel_index++, color.x += step) {
            RGB.color.R = (u8)(255.0f * gammaCorrected(color.x));

            frame_buffer.pixels[pixel_index] = RGB;
        }
    }

    drawRect(&main_rect, border);

    color = *color_control.color;
    setPixelGammaCorrectedColor((&RGB), color);
    R.color.R = RGB.color.R;
    G.color.G = RGB.color.G;
    B.color.B = RGB.color.B;

    fillRect(&color_control.R, R);
    fillRect(&color_control.G, G);
    fillRect(&color_control.B, B);
    fillRect(&color_control.RGB, RGB);
    drawRect(&color_control.R, color_control.is_red_controlled ? border : inactive_border);
    drawRect(&color_control.G, color_control.is_green_controlled ? border : inactive_border);
    drawRect(&color_control.B, color_control.is_blue_controlled ? border : inactive_border);
    drawRect(&color_control.RGB, color_control.is_rgb_controlled ? border : inactive_border);
}


inline void setLightControlsKeyBounds() {
    light_controlls.key_bounds.x_range.min = light_controlls.position.x + CONTROL__LENGTH / 2;
    light_controlls.key_bounds.y_range.min = light_controlls.position.y + CONTROL__SLIDER_LENGTH - light_controlls.key_slider;
    light_controlls.key_bounds.x_range.max = light_controlls.key_bounds.x_range.min + CONTROL__LENGTH;
    light_controlls.key_bounds.y_range.max = light_controlls.key_bounds.y_range.min + CONTROL__THICKNESS;
}

inline void setLightControlsFillBounds() {
    light_controlls.fill_bounds.x_range.min = light_controlls.position.x + CONTROL__LENGTH * 2 + CONTROL__LENGTH / 2;
    light_controlls.fill_bounds.y_range.min = light_controlls.position.y + CONTROL__SLIDER_LENGTH - light_controlls.fill_slider;
    light_controlls.fill_bounds.x_range.max = light_controlls.fill_bounds.x_range.min + CONTROL__LENGTH;
    light_controlls.fill_bounds.y_range.max = light_controlls.fill_bounds.y_range.min + CONTROL__THICKNESS;
}

inline void setLightControlsRimBounds() {
    light_controlls.rim_bounds.x_range.min = light_controlls.position.x + CONTROL__LENGTH * 4 + CONTROL__LENGTH / 2;
    light_controlls.rim_bounds.y_range.min = light_controlls.position.y + CONTROL__SLIDER_LENGTH - light_controlls.rim_slider;
    light_controlls.rim_bounds.x_range.max = light_controlls.rim_bounds.x_range.min + CONTROL__LENGTH;
    light_controlls.rim_bounds.y_range.max = light_controlls.rim_bounds.y_range.min + CONTROL__THICKNESS;
}

inline void bindLightControls(PointLight *key_light, PointLight *fill_light, PointLight *rim_light) {
    light_controlls.key_intensity = &key_light->intensity;
    light_controlls.fill_intensity = &fill_light->intensity;
    light_controlls.rim_intensity = &rim_light->intensity;

    light_controlls.key_slider = (u8)(CONTROL__SLIDER_RANGE * (*light_controlls.key_intensity) * 0.1f);
    light_controlls.fill_slider = (u8)(CONTROL__SLIDER_RANGE * (*light_controlls.fill_intensity) * 0.1f);
    light_controlls.rim_slider = (u8)(CONTROL__SLIDER_RANGE * (*light_controlls.rim_intensity) * 0.1f);
    setLightControlsKeyBounds();
    setLightControlsFillBounds();
    setLightControlsRimBounds();
}

inline void updateKeyLightControl(vec2i movement) {
    updateColorControlComponent(&light_controlls.key_slider, light_controlls.key_intensity, -movement.y, 10);
    setLightControlsKeyBounds();
}

inline void updateFillLightControl(vec2i movement) {
    updateColorControlComponent(&light_controlls.fill_slider, light_controlls.fill_intensity, -movement.y, 10);
    setLightControlsFillBounds();
}

inline void updateRimLightControl(vec2i movement) {
    updateColorControlComponent(&light_controlls.rim_slider, light_controlls.rim_intensity, -movement.y, 10);
    setLightControlsRimBounds();
}

void setLightControlsPosition(u16 position_x, u16 position_y) {
    light_controlls.position.x = position_x;
    light_controlls.position.y = position_y;
    setLightControlsKeyBounds();
    setLightControlsFillBounds();
    setLightControlsRimBounds();
}

void initLightControls(u16 initial_position_x, u16 initial_position_y) {
    light_controlls.is_visible = false;
    light_controlls.is_controlled = false;
    light_controlls.is_key_controlled = false;
    light_controlls.is_fill_controlled = false;
    light_controlls.is_rim_controlled = false;
    setLightControlsPosition(initial_position_x, initial_position_y);
}

void drawLightControls() {
    Pixel key, fill, rim, border, inactive_border;
    border.color = WHITE;
    inactive_border.color = GREY;

    key.color.R = key.color.G = key.color.B    = (u8)(255.0f * (*light_controlls.key_intensity) * 0.1f);
    fill.color.R = fill.color.G = fill.color.B = (u8)(255.0f * (*light_controlls.fill_intensity) * 0.1f);
    rim.color.R = rim.color.G = rim.color.B    = (u8)(255.0f * (*light_controlls.rim_intensity) * 0.1f);

    u32 y_start = light_controlls.position.y + CONTROL__THICKNESS / 2;
    u32 y_end = y_start + CONTROL__SLIDER_LENGTH;
    u32 x = light_controlls.position.x + CONTROL__LENGTH;

    drawVLine2D(y_start, y_end, x, inactive_border);
    drawVLine2D(y_start, y_end, x-1, inactive_border);
    drawVLine2D(y_start, y_end, x+1, inactive_border);

    x += CONTROL__LENGTH*2;

    drawVLine2D(y_start, y_end, x, inactive_border);
    drawVLine2D(y_start, y_end, x-1, inactive_border);
    drawVLine2D(y_start, y_end, x+1, inactive_border);

    x += CONTROL__LENGTH*2;

    drawVLine2D(y_start, y_end, x, inactive_border);
    drawVLine2D(y_start, y_end, x-1, inactive_border);
    drawVLine2D(y_start, y_end, x+1, inactive_border);

    fillRect(&light_controlls.key_bounds,  key);
    fillRect(&light_controlls.fill_bounds, fill);
    fillRect(&light_controlls.rim_bounds,  rim);
    drawRect(&light_controlls.key_bounds,  light_controlls.is_key_controlled  ? border : inactive_border);
    drawRect(&light_controlls.fill_bounds, light_controlls.is_fill_controlled ? border : inactive_border);
    drawRect(&light_controlls.rim_bounds,  light_controlls.is_rim_controlled  ? border : inactive_border);
}



inline void setLightSelectorKeyBounds() {
    light_selector.key_bounds.x_range.min = light_selector.position.x;
    light_selector.key_bounds.y_range.min = light_selector.position.y;
    light_selector.key_bounds.x_range.max = light_selector.key_bounds.x_range.min + CONTROL__LENGTH*2;
    light_selector.key_bounds.y_range.max = light_selector.key_bounds.y_range.min + CONTROL__THICKNESS*2;
}

inline void setLightSelectorFillBounds() {
    light_selector.fill_bounds.x_range.min = light_selector.position.x + CONTROL__LENGTH * 2;
    light_selector.fill_bounds.y_range.min = light_selector.position.y;
    light_selector.fill_bounds.x_range.max = light_selector.fill_bounds.x_range.min + CONTROL__LENGTH*2;
    light_selector.fill_bounds.y_range.max = light_selector.fill_bounds.y_range.min + CONTROL__THICKNESS*2;
}

inline void setLightSelectorRimBounds() {
    light_selector.rim_bounds.x_range.min = light_selector.position.x + CONTROL__LENGTH * 4;
    light_selector.rim_bounds.y_range.min = light_selector.position.y;
    light_selector.rim_bounds.x_range.max = light_selector.rim_bounds.x_range.min + CONTROL__LENGTH*2;
    light_selector.rim_bounds.y_range.max = light_selector.rim_bounds.y_range.min + CONTROL__THICKNESS*2;
}

inline void setLightSelectorAmbientBounds() {
    light_selector.ambient_bounds.x_range.min = light_selector.position.x;
    light_selector.ambient_bounds.y_range.min = light_selector.position.y + CONTROL__THICKNESS*2;
    light_selector.ambient_bounds.x_range.max = light_selector.ambient_bounds.x_range.min + CONTROL__LENGTH*2;
    light_selector.ambient_bounds.y_range.max = light_selector.ambient_bounds.y_range.min + CONTROL__THICKNESS*2;
}

inline void bindLightSelector(PointLight *key_light, PointLight *fill_light, PointLight *rim_light, AmbientLight *ambient_light) {
    light_selector.key_color = &key_light->color;
    light_selector.fill_color = &fill_light->color;
    light_selector.rim_color = &rim_light->color;
    light_selector.ambient_color = &ambient_light->color;

    setLightSelectorKeyBounds();
    setLightSelectorFillBounds();
    setLightSelectorRimBounds();
    setLightSelectorAmbientBounds();
}

void setLightSelectorPosition(u16 position_x, u16 position_y) {
    light_selector.position.x = position_x;
    light_selector.position.y = position_y;
    setLightSelectorKeyBounds();
    setLightSelectorFillBounds();
    setLightSelectorRimBounds();
    setLightSelectorAmbientBounds();
}

void initLightSelector(u16 initial_position_x, u16 initial_position_y) {
    light_selector.is_visible = false;
    light_selector.is_key_selected = true;
    light_selector.is_fill_selected = false;
    light_selector.is_rim_selected = false;
    light_selector.is_ambient_selected = false;
    setLightSelectorPosition(initial_position_x, initial_position_y);
}

void drawLightSelector() {
    Pixel border, inactive_border, text;
    border.color = WHITE;
    inactive_border.color = GREY;
    text.color = GREEN;

    drawRect(&light_selector.key_bounds,  light_selector.is_key_selected  ? border : inactive_border);
    drawRect(&light_selector.fill_bounds, light_selector.is_fill_selected ? border : inactive_border);
    drawRect(&light_selector.rim_bounds,  light_selector.is_rim_selected  ? border : inactive_border);
    drawRect(&light_selector.ambient_bounds,  light_selector.is_ambient_selected  ? border : inactive_border);

    drawText(&frame_buffer, "Key", text.value, light_selector.key_bounds.x_range.min + 5, light_selector.key_bounds.y_range.min + 5);
    drawText(&frame_buffer, "Fill", text.value, light_selector.fill_bounds.x_range.min + 5, light_selector.fill_bounds.y_range.min + 5);
    drawText(&frame_buffer, "Rim", text.value, light_selector.rim_bounds.x_range.min + 5, light_selector.rim_bounds.y_range.min + 5);
    drawText(&frame_buffer, "Amb", text.value, light_selector.ambient_bounds.x_range.min + 5, light_selector.ambient_bounds.y_range.min + 5);
}