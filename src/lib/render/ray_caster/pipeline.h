#pragma once

#include "lib/core/types.h"
#include "lib/core/inputs.h"
#include "lib/nodes/camera.h"
#include "lib/input/controllers.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"


static char* RAY_CASTER_TITLE = "RayCaster";

typedef struct RayCaster {
    Camera2D camera;
} RayCaster;

static RayCaster ray_caster;

void initRayCaster() {
    initCamera2D(&ray_caster.camera);
    ray_caster.camera.transform->position->x = 5;
    ray_caster.camera.transform->position->y = 5;
}

void rayCast() {
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

inline void resetRayDirections2D() {
//    generateRays(
//            ray_caster.ray_directions,
//            ray_tracer.camera.transform->rotation,
//            ray_tracer.camera.focal_length,
//            frame_buffer.width,
//            frame_buffer.height);
}

inline void onZoomRC() {resetRayDirections();}
inline void onOrbitRC() {resetRayDirections();}
inline void onOrientRC() {resetRayDirections();}