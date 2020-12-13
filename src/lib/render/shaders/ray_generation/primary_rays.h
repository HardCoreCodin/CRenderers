#pragma once

#include "lib/core/types.h"

#ifdef __CUDACC__

#include "cuda_runtime.h"

__global__
void setRayDirectionKernel(vec3 start, vec3 right, vec3 down, u16 width, u32 ray_count) {
    initKernel(ray_count, width);

    vec3 ray_direction = start;
    vec3 ray_direction_rcp;

    iscaleVec3(&right, x); iaddVec3(&ray_direction, &right);
    iscaleVec3(&down,  y); iaddVec3(&ray_direction, &down);
    norm3(&ray_direction);

    ray_direction_rcp.x = 1.0f / ray_direction->x;
    ray_direction_rcp.y = 1.0f / ray_direction->y;
    ray_direction_rcp.z = 1.0f / ray_direction->z;

    d_ray_directions[i] = ray_direction;
    d_ray_directions_rcp[i] = ray_direction_rcp;
}
inline void setRayDirectionsGPU(vec3 *start, vec3 *right, vec3 *down) {
    u16 width = frame_buffer.width;
    u16 height = frame_buffer.height;
    u32 count = frame_buffer.size;
    setupKernel(count)

    setRayDirectionKernel<<<blocks, threads>>>(*start, *right, *down, width, count);
    cudaThreadSynchronize();
}
#endif

inline void setRayDirectionsCPU( vec3 *start, vec3 *right, vec3 *down) {
    vec3 *ray_direction = ray_tracer.ray_directions,
         *ray_direction_rcp = ray_tracer.ray_directions_rcp;
    vec3 current = *start;
    u16 width = frame_buffer.width;
    u16 height = frame_buffer.height;

    for (i32 y = height - 1; y > -height; y -= 2) {
        for (i32 x = 1 - width; x < width; x += 2, ray_direction++, ray_direction_rcp++) {
            *ray_direction = current;
            norm3(ray_direction);
            ray_direction_rcp->x = 1.0f / ray_direction->x;
            ray_direction_rcp->y = 1.0f / ray_direction->y;
            ray_direction_rcp->z = 1.0f / ray_direction->z;
            iaddVec3(&current, right);
        }
        iaddVec3(start, down);
        current = *start;
    }
}

inline void generateRayDirections() {
    vec3 start,
         right,
         down,
        *s = &start,
        *r = &right,
        *d = &down,
        *U = main_camera.transform.up_direction,
        *R = main_camera.transform.right_direction,
        *F = main_camera.transform.forward_direction;

    f32 fl = main_camera.focal_length,
        w = (f32)frame_buffer.width,
        h = (f32)frame_buffer.height;

    scaleVec3(F, w * fl, s);
    scaleVec3(R, 1 - w, r);
    scaleVec3(U, h - 2, d);
    iaddVec3(s, r);
    iaddVec3(s, d);

    scaleVec3(R, 2, r);
    scaleVec3(U, -2, d);

#ifdef __CUDACC__
    if (use_GPU) setRayDirectionsGPU(s, r, d);
    else         setRayDirectionsCPU(s, r, d);
#else
    setRayDirectionsCPU(s, r, d);
#endif
}