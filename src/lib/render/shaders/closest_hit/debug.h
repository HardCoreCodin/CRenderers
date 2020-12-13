#pragma once

#include "lib/core/types.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeUV(vec2 uv, vec3* out_color) {
    out_color->x = uv.x * MAX_COLOR_VALUE * 0.5f;
    out_color->y = uv.y * MAX_COLOR_VALUE * 0.5f;
    out_color->z = 0.5;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeDirection(vec3* direction, vec3* out_color) {
    f32 factor = MAX_COLOR_VALUE * 0.5f;
    out_color->x = factor * (direction->x + 1);
    out_color->y = factor * (direction->y + 1);
    out_color->z = factor * (direction->z + 1);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeDepth(f32 distance, vec3* out_color) {
    out_color->x = out_color->y = out_color->z = 4 * MAX_COLOR_VALUE / distance;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeDirectionAndDepth(vec3* direction, f32 distance, vec3* out_color) {
    f32 factor = 4 * MAX_COLOR_VALUE / distance;
    out_color->x = factor * (direction->x + 1);
    out_color->y = factor * (direction->y + 1);
    out_color->z = factor * (direction->z + 1);
}