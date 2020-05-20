#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"

static char* RAY_CASTER_TITLE = "RayCaster";

void onRenderRC(Engine* engine) {
    Camera* camera = engine->active_viewport->controller->camera;
    Pixel* pixel = engine->frame_buffer->pixels;
    Vector2* RO = camera->transform2D->position;
    Vector2* RD = engine->renderers.ray_caster->ray_directions;
}

inline void generateRays2D(Engine* engine) {
    Camera* camera = engine->active_viewport->controller->camera;
    FrameBuffer* frame_buffer = engine->frame_buffer;

    Vector2 right, ray;
    Vector2* rotX = camera->transform2D->rotation->x_axis;
    Vector2* rotY = camera->transform2D->rotation->y_axis;
    Vector2* rays = engine->renderers.ray_caster->ray_directions;
    scale2D(rotX, (1 - (f32)frame_buffer->width) / 2, &right);
    scale2D(rotY, (f32)frame_buffer->height * camera->focal_length / 2, &ray);
    iadd2D(&ray, &right);
    right = *rotX;
    for (u16 w = 0; w < frame_buffer->width; w++) {
        scale2D(&ray, 1 / sqrtf(ray.x*ray.x + ray.y*ray.y), rays++);
        iadd2D(&ray, &right);
    }
}

void onZoomRC(Engine* engine) { generateRays2D(engine); }
void onRotateRC(Engine* engine) { generateRays2D(engine);}
void onMoveRC(Engine* engine) {}
void onResizeRC(Engine* engine) { generateRays2D(engine); }

RayCaster* createRayCaster(Engine* engine) {
    RayCaster* ray_caster = Alloc(RayCaster);
    ray_caster->renderer.title = RAY_CASTER_TITLE;
    ray_caster->rays_per_pixel = 1;
    ray_caster->ray_count = ray_caster->rays_per_pixel * engine->frame_buffer->width * engine->frame_buffer->height;
    ray_caster->ray_directions = AllocN(Vector2, ray_caster->ray_count);
    return ray_caster;
}