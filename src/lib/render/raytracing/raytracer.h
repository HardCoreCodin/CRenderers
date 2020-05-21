#pragma once

#include "lib/core/types.h"
#include "lib/shapes/line.h"
#include "lib/input/keyboard.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"

#include "lib/render/raytracing/shaders/closest_hit/normal.h"
#include "lib/render/raytracing/shaders/intersection/ray_sphere.h"
#include "lib/render/raytracing/shaders/generation/ray_generation.h"

static char* RAY_TRACER_TITLE = "RayTrace";

void onRenderRT(Engine* engine) {
    RayTracer* ray_tracer = engine->renderers.ray_tracer;
    FrameBuffer* frame_buffer = engine->frame_buffer;
    Pixel* pixel = frame_buffer->pixels;
    Vector3* ray_direction = ray_tracer->ray_directions;
    Sphere* sphere = engine->scene->spheres;
    u8 sphere_count = engine->scene->sphere_count;

    for (u32 i = 0; i < frame_buffer->size; i++)
        if (rayIntersectsWithSpheres(ray_tracer->closest_hit, ray_direction++, sphere, sphere_count))
            shadeClosestHitByNormal(ray_tracer->closest_hit, pixel++);
//            shadeRayByDirection(ray_direction++, pixel++);
        else
            (pixel++)->value = 0;
}

void generateRaysRT(Engine* engine) {
    generateRayDirections(
            engine->renderers.ray_tracer->ray_directions,
            engine->active_viewport->controller->camera->focal_length,
            engine->frame_buffer->width,
            engine->frame_buffer->height);
}

void onResizeRT(Engine* engine) {
    generateRaysRT(engine);
}

void onZoomRT(Engine* engine) {
    generateRaysRT(engine);
    engine->active_viewport->controller->changed.fov = false;
}

void onRotateRT(Engine* engine) {
    Controller* controller = engine->active_viewport->controller;
    RayTracer* ray_tracer = engine->renderers.ray_tracer;

    transposeMatrix3D(controller->camera->transform->rotation, ray_tracer->inverted_camera_rotation);
    controller->changed.orientation = false;
    controller->changed.position = true;
}

void onMoveRT(Engine* engine) {
    Controller* controller = engine->active_viewport->controller;
    RayTracer* ray_tracer = engine->renderers.ray_tracer;
    Vector3* camera_position = controller->camera->transform->position;
    Sphere* sphere = engine->scene->spheres;
    const Sphere* last_sphere = sphere + engine->scene->sphere_count;
    while (sphere != last_sphere) {
        sub3D(sphere->world_position, camera_position, sphere->view_position);
        imul3D(sphere->view_position, ray_tracer->inverted_camera_rotation);
        sphere++;
    }

    controller->changed.position = false;
}

RayTracer* createRayTracer(Engine* engine) {
    RayTracer* ray_tracer = Alloc(RayTracer);
    ray_tracer->renderer.title = RAY_TRACER_TITLE;
    ray_tracer->renderer.on.zoom = onZoomRT;
    ray_tracer->renderer.on.move = onMoveRT;
    ray_tracer->renderer.on.rotate = onRotateRT;
    ray_tracer->renderer.on.resize = onResizeRT;
    ray_tracer->renderer.on.render = onRenderRT;
    ray_tracer->rays_per_pixel = 1;
    ray_tracer->ray_count = engine->frame_buffer->width * engine->frame_buffer->height * ray_tracer->rays_per_pixel;
    ray_tracer->ray_directions = AllocN(Vector3, ray_tracer->ray_count);
    ray_tracer->closest_hit = Alloc(RayHit);
    ray_tracer->inverted_camera_rotation = createMatrix3x3();
    return ray_tracer;
}