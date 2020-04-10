#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"
#include "lib/render/raytracing/shaders/closest_hit/normal.h"
#include "lib/render/raytracing/shaders/intersection/ray_sphere.h"
#include "lib/render/raytracing/shaders/generation/ray_generation.h"

#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"

typedef struct {
    Renderer base;
    Camera3D camera;
    bool in_fps_mode;
    u32 ray_count;
    u8 rays_per_pixel;
    RayHit* closest_hit;
    Vector3 *ray_directions;
} RayTracer;

static RayTracer ray_tracer;

void rayTrace() {
    Pixel* pixel = (Pixel*)frame_buffer.pixels;
    Vector3* ray_direction = ray_tracer.ray_directions;

    for (u32 i = 0; i < frame_buffer.size; i++)
        if (rayIntersectsWithSpheres(ray_tracer.closest_hit, ray_direction++))
            shadeClosestHitByNormal(ray_tracer.closest_hit, pixel++);
//            shadeRayByDirection(ray_direction++, pixel++);
        else
            (pixel++)->value = 0;
}

void generateRaysRT() {
    generateRayDirections(
            ray_tracer.ray_directions,
            ray_tracer.camera.focal_length,
            frame_buffer.width,
            frame_buffer.height);
}

void switchControllerRT() {
    ray_tracer.in_fps_mode = !ray_tracer.in_fps_mode;
    ray_tracer.base.controller = ray_tracer.in_fps_mode ? &fps.controller : &orb.controller;
    setControllerModeInHUD(ray_tracer.in_fps_mode);
}

void onResizedRT() {
    generateRaysRT();
    updateHUDDimensions();
}

void onZoomedRT(Controller* controller) {
    generateRaysRT();
    controller->zoomed = false;
}

void onRotatedRT(Controller* controller) {
    transposeMatrix3D(ray_tracer.camera.transform->rotation, ray_tracer.camera.transform->rotation_inverted);
    controller->rotated = false;
    controller->moved = true;
}
void onMovedRT(Controller* controller) {
    Vector3* camera_position = ray_tracer.camera.transform->position;
    Matrix3x3* inverted_camera_rotation = ray_tracer.camera.transform->rotation_inverted;

    Sphere *sphere = scene.spheres;
    for (u8 i = 0; i < scene.sphere_count; i++) {
        sub3D(sphere->world_position, camera_position, sphere->view_position);
        imul3D(sphere->view_position, inverted_camera_rotation);
        sphere++;
    }

    controller->moved = false;
}

void initRayTracer() {
    ray_tracer.base.title = "RayTrace";
    ray_tracer.base.on.render = rayTrace;
    ray_tracer.base.on.resized = onResizedRT;
    ray_tracer.base.on.double_clicked = switchControllerRT;
    ray_tracer.base.controller = &orb.controller;

    ray_tracer.rays_per_pixel = 1;
    ray_tracer.ray_count = frame_buffer.width * frame_buffer.height * ray_tracer.rays_per_pixel;
    ray_tracer.ray_directions = (Vector3*)allocate(sizeof(Vector3) * ray_tracer.ray_count);

    initCamera3D(&ray_tracer.camera);
    ray_tracer.camera.transform->position->x = 5;
    ray_tracer.camera.transform->position->y = 5;
    ray_tracer.camera.transform->position->z = -10;
    onMovedRT(&orb.controller);

    ray_tracer.closest_hit = (RayHit*)allocate(sizeof(RayHit));

    initFpsController(&ray_tracer.camera, onZoomedRT, onMovedRT, onRotatedRT);
    initOrbController(&ray_tracer.camera, onMovedRT, onMovedRT, onRotatedRT);
}