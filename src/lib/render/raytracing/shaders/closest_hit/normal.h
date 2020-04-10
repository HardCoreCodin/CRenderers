#pragma once

#include "lib/core/types.h"

#define MAX_COLOR_VALUE 0xFF

void shadeClosestHitByNormal(RayHit* closestHit, Pixel* pixel) {
    f32 factor = 4 * MAX_COLOR_VALUE / closestHit->distance;
    f32 R = factor * (closestHit->normal.x + 1);
    f32 G = factor * (closestHit->normal.y + 1);
    f32 B = factor * (closestHit->normal.z + 1);

    pixel->color.R = R > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)R;
    pixel->color.G = G > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)G;
    pixel->color.B = B > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)B;
}

void shadeRayByDirection(Vector3* ray_direction, Pixel* pixel) {
    f32 factor = 0.5f * MAX_COLOR_VALUE;
    f32 R = factor * (ray_direction->x + 1);
    f32 G = factor * (ray_direction->y + 1);
    f32 B = factor * (ray_direction->z + 1);

    pixel->color.R = R > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)R;
    pixel->color.G = G > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)G;
    pixel->color.B = B > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)B;
}