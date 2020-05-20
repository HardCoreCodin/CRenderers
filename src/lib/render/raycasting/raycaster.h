#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"

typedef struct RayCaster {
    Renderer renderer;
    u32 ray_count;
    u8 rays_per_pixel;
    Vector2* ray_directions;
} RayCaster;

RayCaster ray_caster;

void onRenderRC(Viewport* viewport) {
    u32* pixel = frame_buffer.pixels;
    Vector2* RO = viewport->controller->camera->transform2D->position;
    Vector2* RD = ray_caster.ray_directions;
}

inline void generateRays2D(Camera* camera) {
    Vector2 right, ray;
    Vector2* rotX = camera->transform2D->rotation->x_axis;
    Vector2* rotY = camera->transform2D->rotation->y_axis;
    Vector2* rays = ray_caster.ray_directions;
    scale2D(rotX, (1 - (f32)frame_buffer.width) / 2, &right);
    scale2D(rotY, (f32)frame_buffer.height * camera->focal_length / 2, &ray);
    iadd2D(&ray, &right);
    right = *rotX;
    for (u16 w = 0; w < frame_buffer.width; w++) {
        scale2D(&ray, 1 / sqrtf(ray.x*ray.x + ray.y*ray.y), rays++);
        iadd2D(&ray, &right);
    }
}

void onZoomRC(Viewport* viewport) {generateRays2D(viewport->controller->camera);}
void onRotateRC(Viewport* viewport) {generateRays2D(viewport->controller->camera);}
void onMoveRC(Viewport* viewport) {}
void onResizeRC(Viewport* viewport) {generateRays2D(viewport->controller->camera);}

void initRayCaster(Scene* scene) {
    ray_caster.renderer.title = "RayCaster";
    ray_tracer.renderer.scene = scene;
    ray_caster.rays_per_pixel = 1;
    ray_caster.ray_count = ray_caster.rays_per_pixel * frame_buffer.width * frame_buffer.height;
    ray_caster.ray_directions = AllocN(Vector2, ray_caster.ray_count);
}