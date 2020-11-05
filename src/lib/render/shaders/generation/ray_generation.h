#pragma once

#include <math.h>
#include "lib/core/types.h"

inline void generateRayDirections(vec3* ray_direction, f32 focal_length, u16 width, u16 height) {
    f32 z = (f32)width * focal_length;
    f32 z2 = z * z;
    f32 factor, y2_plus_z2;

    for (i32 y = height - 1; y > -height; y -= 2) {
        y2_plus_z2 = (f32)y*y + z2;
        for (i32 x = 1 - width; x < width; x += 2) {
            factor = 1 / sqrtf((f32)x*x + y2_plus_z2);
            ray_direction->x = x * factor;
            ray_direction->y = y * factor;
            ray_direction->z = z * factor;
            ray_direction++;
        }
    }
}

