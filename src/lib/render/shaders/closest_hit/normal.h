#pragma once

#include "lib/core/types.h"

#define MAX_COLOR_VALUE 0xFF

inline void shadeClosestHitByNormal1(RayHit* closestHit, Pixel* pixel) {
    f32 factor = 4 * MAX_COLOR_VALUE / closestHit->distance;
    f32 R = factor * (closestHit->normal.x + 1);
    f32 G = factor * (closestHit->normal.y + 1);
    f32 B = factor * (closestHit->normal.z + 1);
    if (R > MAX_COLOR_VALUE) R = MAX_COLOR_VALUE;
    if (G > MAX_COLOR_VALUE) G = MAX_COLOR_VALUE;
    if (B > MAX_COLOR_VALUE) B = MAX_COLOR_VALUE;

    pixel->value = ((u8)R << 16) + ((u8)G << 8) + (u8)B;
//    pixel->color.G = G;
//    pixel->color.B = B;
}

void shadeClosestHitByNormal(RayHit* closestHit, Pixel* pixel) {
    f32 factor = 4 * MAX_COLOR_VALUE / closestHit->distance;
    f32 R = factor * (closestHit->normal.x + 1);
    f32 G = factor * (closestHit->normal.y + 1);
    f32 B = factor * (closestHit->normal.z + 1);

    pixel->color.R = R > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)R;
    pixel->color.G = G > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)G;
    pixel->color.B = B > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)B;
}

void shadeRayByDirection(vec3* ray_direction, Pixel* pixel) {
    f32 factor = 0.5f * MAX_COLOR_VALUE;
    f32 R = factor * (ray_direction->x + 1);
    f32 G = factor * (ray_direction->y + 1);
    f32 B = factor * (ray_direction->z + 1);

    pixel->color.R = R > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)R;
    pixel->color.G = G > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)G;
    pixel->color.B = B > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)B;
}