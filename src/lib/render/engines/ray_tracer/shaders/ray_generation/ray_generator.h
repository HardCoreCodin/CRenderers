#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"

void generateRayDirections(Vector3* ray_directions, f32 focal_length, u16 width, u16 height) {
    f32 norm_width = 2 / focal_length;
    f32 pixel_size = norm_width / (f32)width;
    f32 norm_height = pixel_size * (f32)height;
    f32 x_start = (pixel_size  - norm_width) / 2;
    f32 y_start = (norm_height - pixel_size) / 2;

    f32 x, x2, y2, ray_direction_length;
    f32 y = y_start;

    for (u16 h = 0; h < height; h++) {
        x = x_start;
        y2 = y * y;

        for (u16 w = 0; w < width; w++) {
            x2 = x * x;

            ray_direction_length = sqrtf(x2 + y2 + 1);

            ray_directions->x = x / ray_direction_length;
            ray_directions->y = y / ray_direction_length;
            ray_directions->z = 1 / ray_direction_length;
            ray_directions++;

            x += pixel_size;
        }

        y -= pixel_size;
    }
}

void generateRayDirectionsRat(Vector3* ray_directions, f32 focal_length, u16 width, u16 height) {
    f32 norm_width = 1 / focal_length;
    f32 pixel_size = norm_width / (f32)width;
    f32 norm_height = pixel_size * (f32)height;
    f32 x_start = (pixel_size  - norm_width) / 2;
    f32 y_start = (norm_height - pixel_size) / 2;

    f32 r, s = y_start;
    f32 r2, s2, f;

    for (u16 y = 0; y < height; y++) {
        r = x_start;

        for (u16 x = 0; x < width; x++) {
            r2 = r * r;
            s2 = s * s;
            f = 1 / (1 + r2 + s2);

            ray_directions->x = 2 * r * f;
            ray_directions->y = 2 * s * f;
            ray_directions->z = (1 - r2 - s2) * f;
            ray_directions++;

            r += pixel_size;
        }

        s -= pixel_size;
    }
}