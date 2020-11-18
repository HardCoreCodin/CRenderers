#pragma once

#include "lib/core/types.h"

inline void generateRayDirections(vec3* ray_direction, Camera* camera, u16 width, u16 height) {
    vec3 right, down, start;
    scaleVec3(camera->transform.forward_direction, (f32)width * camera->focal_length, &start);
    scaleVec3(camera->transform.right_direction, 1.0f - (f32)width, &right);
    scaleVec3(camera->transform.up_direction, (f32)height - 1.0f - 1, &down);
    iaddVec3(&start, &right);
    iaddVec3(&start, &down);
    vec3 current = start;

    scaleVec3(camera->transform.right_direction, 2, &right);
    scaleVec3(camera->transform.up_direction, -2, &down);

    for (i32 y = height - 1; y > -height; y -= 2) {
        for (i32 x = 1 - width; x < width; x += 2) {
            *ray_direction = current;
            norm3(ray_direction++);
            iaddVec3(&current, &right);
        }
        iaddVec3(&start, &down);
        current = start;
    }
}