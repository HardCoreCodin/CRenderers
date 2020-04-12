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

void rayCast(Controller* controller) {
    u32* pixel = frame_buffer.pixels;
    Vector2* RO = controller->camera->transform2D->position;
    Vector2* RD = ray_caster.ray_directions;
}

inline void generateRays2D(Camera* camera) {
    Vector2 right, ray;
    Vector2* rotX = camera->transform2D->rotation.x_axis;
    Vector2* rotY = camera->transform2D->rotation.y_axis;
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

void zoomRC(Controller* controller) {generateRays2D(controller->camera);}
void rotateRC(Controller* controller) {generateRays2D(controller->camera);}
void moveRC(Controller* controller) {}
void resizeC(Controller* controller) {generateRays2D(controller->camera);}

void initRayCaster(Engine* engine) {
    ray_caster.renderer.title = "RayCaster";
    ray_caster.renderer.render = rayCast;
    ray_caster.renderer.resize = resizeC;
    ray_caster.renderer.move = moveRC;
    ray_caster.renderer.zoom = zoomRC;
    ray_caster.renderer.rotate = rotateRC;
    ray_caster.rays_per_pixel = 1;
    ray_caster.ray_count = ray_caster.rays_per_pixel * frame_buffer.width * frame_buffer.height;
    ray_caster.ray_directions = AllocN(Vector2, ray_caster.ray_count);

    engine->scene.camera.transform2D->position->x = 5;
    engine->scene.camera.transform2D->position->y = 5;
    moveRC(&orb.controller);
}