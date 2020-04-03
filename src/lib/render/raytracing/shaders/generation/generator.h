#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"

void generateRays(Vector3* rays, Matrix3x3* rotation, f32 focal_length, u16 width, u16 height) {
    f32 ray_y2;
    Vector3 up, right, start, ray;
    scale3D(&rotation->i, (1 - (f32)width) / 2, &right);
    scale3D(&rotation->j, ((f32)height - 1) / 2, &up);
    scale3D(&rotation->k, (f32)width * focal_length / 2, &start);
    iadd3D(&start, &right);
    iadd3D(&start, &up);
    up = rotation->j;
    right = rotation->i;

    for (u16 h = 0; h < height; h++) {
        ray = start;
        ray_y2 = ray.y * ray.y;
        for (u16 w = 0; w < width; w++) {
            scale3D(&ray, 1 / sqrtf(ray.x*ray.x + ray_y2 + ray.z*ray.z), rays++);
            iadd3D(&ray, &right);
        }
        isub3D(&start, &up);
    }
}
//
//void generateRayDirectionsRat(Vector3* ray_directions, f32 focal_length, u16 width, u16 height) {
//    f32 norm_width = 1 / focal_length;
//    f32 pixel_size = norm_width / (f32)width;
//    f32 norm_height = pixel_size * (f32)height;
//    f32 x_start = (pixel_size  - norm_width) / 2;
//    f32 y_start = (norm_height - pixel_size) / 2;
//
//    f32 r, s = y_start;
//    f32 r2, s2, f;
//
//    for (u16 y = 0; y < height; y++) {
//        r = x_start;
//
//        for (u16 x = 0; x < width; x++) {
//            r2 = r * r;
//            s2 = s * s;
//            f = 1 / (1 + r2 + s2);
//
//            ray_directions->x = 2 * r * f;
//            ray_directions->y = 2 * s * f;
//            ray_directions->z = (1 - r2 - s2) * f;
//            ray_directions++;
//
//            r += pixel_size;
//        }
//
//        s -= pixel_size;
//    }
//}