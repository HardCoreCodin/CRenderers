#pragma once

#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"

typedef struct RayHit {
    Vector3 position, normal;
    f32 distance;
} RayHit;

typedef struct RayTracer {
    Camera3D camera;
    u32 ray_count;
    u8 rays_per_pixel;
    Vector3* ray_directions;
} RayTracer;