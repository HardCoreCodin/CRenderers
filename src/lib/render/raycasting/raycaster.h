#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/camera.h"
#include "lib/input/controllers.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"


static char* RAY_CASTER_TITLE = "RayCaster";

typedef struct RayCaster {
    Camera2D camera;
    u32 ray_count;
    u8 rays_per_pixel;
    Vector2* ray_directions;
} RayCaster;

static RayCaster ray_caster;

void initRayCaster() {
    ray_caster.rays_per_pixel = 1;
    ray_caster.ray_count = frame_buffer.width * frame_buffer.height * ray_caster.rays_per_pixel;
    ray_caster.ray_directions = (Vector2*)allocate(sizeof(Vector2) * ray_caster.ray_count);

    initCamera2D(&ray_caster.camera);
    ray_caster.camera.transform->position->x = 5;
    ray_caster.camera.transform->position->y = 5;
}

void rayCast() {
    u32* pixel = frame_buffer.pixels;
    Vector2* RO = ray_caster.camera.transform->position;
    Vector2* RD = ray_caster.ray_directions;
}

inline void generateRays2D() {
    Vector2 right, ray;
    Vector2* rotX = &ray_caster.camera.transform->rotation->i;
    Vector2* rotY = &ray_caster.camera.transform->rotation->j;
    Vector2* rays = ray_caster.ray_directions;
    scale2D(rotX, (1 - (f32)frame_buffer.width) / 2, &right);
    scale2D(rotY, (f32)frame_buffer.height * renderer.camera.focal_length / 2, &ray);
    iadd2D(&ray, &right);
    right = *rotX;
    for (u16 w = 0; w < frame_buffer.width; w++) {
        scale2D(&ray, 1 / sqrtf(ray.x*ray.x + ray.y*ray.y), rays++);
        iadd2D(&ray, &right);
    }
}

inline void onZoomRC() {generateRays2D();}
inline void onOrbitRC() {generateRays2D();}
inline void onOrientRC() {generateRays2D();}
inline void onResizeRC() {generateRays2D();}