#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

const float COLOR_FACTOR = 0xFF / 2.0f;

Color color;

void shadeByNormal(u32* pixel, Vector3* normal) {
    color.R = (u8)((normal->x + 1) * COLOR_FACTOR);
    color.G = (u8)((normal->y + 1) * COLOR_FACTOR);
    color.B = (u8)((normal->z + 1) * COLOR_FACTOR);
    *pixel = color.value;
}