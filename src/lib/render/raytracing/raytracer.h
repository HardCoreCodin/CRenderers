#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/camera.h"
#include "lib/input/controllers.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"
#include "lib/render/raytracing/raytrace_types.h"
#include "lib/render/raytracing/shaders/closest_hit/normal.h"
#include "lib/render/raytracing/shaders/intersection/ray_sphere.h"

static char* RAY_TRACER_TITLE = "RayTrace";

RayTracer ray_tracer;
RayHit* closest_hit;

void initRayTracer() {
    ray_tracer.rays_per_pixel = 1;
    ray_tracer.ray_count = frame_buffer.width * frame_buffer.height * ray_tracer.rays_per_pixel;
    ray_tracer.ray_directions = (Vector3*)allocate(sizeof(Vector3) * ray_tracer.ray_count);

    initCamera3D(&ray_tracer.camera);
    ray_tracer.camera.transform->position->x = 5;
    ray_tracer.camera.transform->position->y = 5;
    ray_tracer.camera.transform->position->z = -10;

    closest_hit = (RayHit*)allocate(sizeof(RayHit));
}

void rayTrace() {
    Pixel* pixel = (Pixel*)frame_buffer.pixels;
    Vector3* RO = ray_tracer.camera.transform->position;
    Vector3* RD = ray_tracer.ray_directions;

    for (u32 i = 0; i < frame_buffer.size; i++)
        if (rayIntersectsWithSpheres(closest_hit, RO, RD++))
            shadeClosestHitByNormal(closest_hit, pixel++);
        else
            (pixel++)->value = 0;
}

inline void generateRays3D() {
    f32 ray_y2;
    Vector3 up, right, start, ray;
    Vector3* rotX = &ray_tracer.camera.transform->rotation->i;
    Vector3* rotY = &ray_tracer.camera.transform->rotation->j;
    Vector3* rotZ = &ray_tracer.camera.transform->rotation->k;
    Vector3* rays = ray_tracer.ray_directions;
    scale3D(rotX, (1 - (f32)frame_buffer.width) / 2, &right);
    scale3D(rotY, ((f32)frame_buffer.height - 1) / 2, &up);
    scale3D(rotZ, (f32)frame_buffer.width * ray_tracer.camera.focal_length / 2, &start);
    iadd3D(&start, &right);
    iadd3D(&start, &up);
    up = *rotY;
    right = *rotX;

    for (u16 h = 0; h < frame_buffer.height; h++) {
        ray = start;
        ray_y2 = ray.y * ray.y;
        for (u16 w = 0; w < frame_buffer.width; w++) {
            scale3D(&ray, 1 / sqrtf(ray.x*ray.x + ray_y2 + ray.z*ray.z), rays++);
            iadd3D(&ray, &right);
        }
        isub3D(&start, &up);
    }
}

inline void onZoomRT() {generateRays3D();}
inline void onOrbitRT() {generateRays3D();}
inline void onOrientRT() {generateRays3D();}
inline void onResizeRT() {generateRays3D();}

//
//void generateRays3D(Vector3* ray_directions, f32 focal_length, u16 width, u16 height) {
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