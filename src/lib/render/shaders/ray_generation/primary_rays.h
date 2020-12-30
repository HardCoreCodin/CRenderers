#pragma once

#include "lib/core/types.h"

#ifdef __CUDACC__

#include "cuda_runtime.h"

__global__
void setRayDirectionKernel(vec3 start, vec3 right, vec3 down) {
    initKernel();

    vec3 ray_direction = start;

    iscaleVec3(&right, x); iaddVec3(&ray_direction, &right);
    iscaleVec3(&down,  y); iaddVec3(&ray_direction, &down);
    norm3(&ray_direction);

//    d_ray_directions[i] = ray_direction;
}
inline void setRayDirectionsGPU(vec3 *start, vec3 *right, vec3 *down) {
    setupKernel()

    setRayDirectionKernel<<<blocks, threads>>>(*start, *right, *down);
}
#endif

inline void setRayDirectionsCPU(vec3 *start, vec3 *right, vec3 *down) {
    vec3 *ray_direction = ray_tracer.ray_directions;
    vec3 current = *start;
    u16 width = frame_buffer.dimentions.width;
    u16 height = frame_buffer.dimentions.height;

    for (i32 y = height - 1; y > -height; y -= 2) {
        for (i32 x = 1 - width; x < width; x += 2, ray_direction++) {
            *ray_direction = current;
            norm3(ray_direction);
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
        w = (f32)frame_buffer.dimentions.width,
        h = (f32)frame_buffer.dimentions.height;

    scaleVec3(F, w * fl, s);
    scaleVec3(R, 1 - w, r);
    scaleVec3(U, h - 2, d);
    iaddVec3(s, r);
    iaddVec3(s, d);

    scaleVec3(R, 2, r);
    scaleVec3(U, -2, d);

//#ifdef __CUDACC__
//    setRayDirectionsCPU(s, r, d);
//    gpuErrchk(cudaMemcpyToSymbol(d_ray_directions, ray_tracer.ray_directions, sizeof(vec3) * frame_buffer.dimentions.width_times_height, 0, cudaMemcpyHostToDevice));
//    if (use_GPU) setRayDirectionsGPU(s, r, d);
//    else         setRayDirectionsCPU(s, r, d);
//#else
//    setRayDirectionsCPU(s, r, d);
//#endif
}