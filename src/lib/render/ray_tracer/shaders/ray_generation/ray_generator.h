#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"

void generateRays(Vector3* ray_directions, Matrix3x3* rotation, f32 focal_length, u16 width, u16 height) {
    f32 norm_width = 2 / focal_length;
    f32 pixel_size = norm_width / (f32)width;
    f32 norm_height = pixel_size * (f32)height;

    f32 x = (pixel_size  - norm_width) / 2;
    f32 y = (norm_height - pixel_size) / 2;

    f32 sx = x*rotation->m11 + y*rotation->m21 + rotation->m31;
    f32 sy = x*rotation->m12 + y*rotation->m22 + rotation->m32;
    f32 sz = x*rotation->m13 + y*rotation->m23 + rotation->m33;

    f32 x_step_per_x = rotation->i.x * pixel_size;
    f32 z_step_per_x = rotation->i.z * pixel_size;

    f32 x_step_per_y = -rotation->j.x * pixel_size;
    f32 y_step_per_y = -rotation->j.y * pixel_size;
    f32 z_step_per_y = -rotation->j.z * pixel_size;

    f32 ray_direction_length, z, y2;
    y = sy;

    for (u16 h = 0; h < height; h++) {
        y2 = y * y;
        x = sx;
        z = sz;

        for (u16 w = 0; w < width; w++) {
            ray_direction_length = sqrtf(x*x + y2 + z*z);

            ray_directions->x = x / ray_direction_length;
            ray_directions->y = y / ray_direction_length;
            ray_directions->z = z / ray_direction_length;
            ray_directions++;

            x += x_step_per_x;
            z += z_step_per_x;
        }

        y += y_step_per_y;
        sx += x_step_per_y;
        sz += z_step_per_y;
    }
}
//
//void generateRayDirections(Vector3* ray_directions, f32 focal_length, u16 width, u16 height) {
//    f32 norm_width = 2 / focal_length;
//    f32 pixel_size = norm_width / (f32)width;
//    f32 norm_height = pixel_size * (f32)height;
//    f32 x_start = (pixel_size  - norm_width) / 2;
//    f32 y_start = (norm_height - pixel_size) / 2;
//
//    f32 x, x2, y2, ray_direction_length;
//    f32 y = y_start;
//
//    for (u16 h = 0; h < height; h++) {
//        x = x_start;
//        y2 = y * y;
//
//        for (u16 w = 0; w < width; w++) {
//            x2 = x * x;
//
//            ray_direction_length = sqrtf(x2 + y2 + 1);
//
//            ray_directions->x = x / ray_direction_length;
//            ray_directions->y = y / ray_direction_length;
//            ray_directions->z = 1 / ray_direction_length;
//            ray_directions++;
//
//            x += pixel_size;
//        }
//
//        y -= pixel_size;
//    }
//}
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