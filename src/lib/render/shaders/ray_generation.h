#pragma once

#include "lib/core/types.h"

#ifdef __CUDACC__

#include "cuda_runtime.h"

__global__
void setRayDirectionKernel(vec3 ray_direction, vec3 right, vec3 down, u16 width, u32 count) {
    u32 i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= count) return;

    f32 x = i % width;
    f32 y = i / width;

    iscaleVec3(&right, x); iaddVec3(&ray_direction, &right);
    iscaleVec3(&down,  y); iaddVec3(&ray_direction, &down);
    norm3(&ray_direction);
    d_ray_directions[i] = ray_direction;
}
inline void setRayDirectionsGPU(vec3 *h_ray_directions, u16 width, u16 height, vec3 start, vec3 right, vec3 down) {
    u32 count = (u32)width * (u32)height;
    u16 threads = CUDA_MAX_THREADS;
    u16 blocks  = count / CUDA_MAX_THREADS;
    if (count < CUDA_MAX_THREADS) {
        threads = count;
        blocks = 1;
    }

    setRayDirectionKernel<<<blocks, threads>>>(start, right, down, width, count);
    cudaThreadSynchronize();
    cudaMemcpyFromSymbol(
            h_ray_directions,
            d_ray_directions,
            sizeof(vec3) * width * height, 0,
            cudaMemcpyDeviceToHost);
}
#endif


inline void setRayDirectionsCPU(vec3 *ray_direction, u16 width, u16 height, vec3 start, vec3 right, vec3 down) {
    vec3 current = start;
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

inline void generateRayDirections(vec3* ray_direction, Camera* camera, u16 width, u16 height) {
    vec3 right, down, start;
    scaleVec3(camera->transform.forward_direction, (f32)width * camera->focal_length, &start);
    scaleVec3(camera->transform.right_direction, 1.0f - (f32)width, &right);
    scaleVec3(camera->transform.up_direction, (f32)height - 1.0f - 1, &down);
    iaddVec3(&start, &right);
    iaddVec3(&start, &down);

    scaleVec3(camera->transform.right_direction, 2, &right);
    scaleVec3(camera->transform.up_direction, -2, &down);

#ifdef __CUDACC__
    setRayDirectionsGPU(ray_direction, width, height, start, right, down);
#else
    setRayDirectionsCPU(ray_direction, width, height, start, right, down);
#endif
}