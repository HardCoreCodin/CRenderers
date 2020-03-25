#pragma once

#include "lib/core/types.h"
#include "lib/render/engines/ray_tracer/ray.h"

const float COLOR_FACTOR = 0xFF / 2.0f;

Color color;

void shadeByNormal(u32* pixel, RayHit* hit) {
    color.R = (u8)((hit->normal.x + 1) * COLOR_FACTOR);
    color.G = (u8)((hit->normal.y + 1) * COLOR_FACTOR);
    color.B = (u8)((hit->normal.z + 1) * COLOR_FACTOR);
    *pixel = color.value;
}