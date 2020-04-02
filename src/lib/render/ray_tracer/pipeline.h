#pragma once

#include "lib/core/types.h"
#include "lib/core/inputs.h"
#include "lib/nodes/camera.h"
#include "lib/input/controllers.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"
#include "lib/render/ray_tracer/core/ray.h"
#include "lib/render/ray_tracer/shaders/closest_hit/normal.h"
#include "lib/render/ray_tracer/shaders/ray_generation/ray_generator.h"
#include "lib/render/ray_tracer/shaders/intersection/ray_sphere.h"

static char* RAY_TRACER_TITLE = "RayTrace";

typedef struct RayTracer {
    Camera3D camera;
    Ray3D ray;
    u32 ray_count;
    u8 rays_per_pixel;
    Vector3* ray_directions;
} RayTracer;

static RayTracer ray_tracer;

void initRayTracer() {
    ray_tracer.rays_per_pixel = 1;
    ray_tracer.ray_count = frame_buffer.width * frame_buffer.height * ray_tracer.rays_per_pixel;
    ray_tracer.ray_directions = (Vector3*)allocate(sizeof(Vector3) * ray_tracer.ray_count);

    initCamera3D(&ray_tracer.camera);
    ray_tracer.camera.transform->position->x = 5;
    ray_tracer.camera.transform->position->y = 5;
    ray_tracer.camera.transform->position->z = -10;

    initRay(&ray_tracer.ray);
    ray_tracer.ray.origin = ray_tracer.camera.transform->position;
    ray_tracer.ray.direction = ray_tracer.ray_directions;
}

void rayTrace() {
    u32* pixel = frame_buffer.pixels;
    Vector3* RO = ray_tracer.ray.origin;
    Vector3* RD = ray_tracer.ray_directions;
    Vector3* P = ray_tracer.ray.hit_position;
    Vector3* N = ray_tracer.ray.hit_normal;

    for (u32 i = 0; i < frame_buffer.size; i++) {
        if (rayIntersectsWithSpheres(RO, RD++, P, N))
            shadeByNormal(pixel++, ray_tracer.ray.hit_normal);
        else
            *pixel++ = 0;
    }
}

inline void resetRayDirections() {
    generateRays(
            ray_tracer.ray_directions,
            ray_tracer.camera.transform->rotation,
            ray_tracer.camera.focal_length,
            frame_buffer.width,
            frame_buffer.height);
}

inline void onZoomRT() {resetRayDirections();}
inline void onOrbitRT() {resetRayDirections();}
inline void onOrientRT() {resetRayDirections();}