#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"

#include "lib/render/raytracing/shaders/closest_hit/normal.h"
#include "lib/render/raytracing/shaders/intersection/ray_sphere.h"
#include "lib/render/raytracing/shaders/generation/ray_generation.h"

typedef struct {
    Renderer renderer;
    u32 ray_count;
    u8 rays_per_pixel;
    Scene* scene;
    RayHit* closest_hit;
    Vector3 *ray_directions;
    Matrix3x3 inverted_camera_rotation;
} RayTracer;

RayTracer ray_tracer;

void rayTrace(Controller* controller) {
    Pixel* pixel = (Pixel*)frame_buffer.pixels;
    Vector3* ray_direction = ray_tracer.ray_directions;
    Sphere* sphere = ray_tracer.scene->spheres;
    u8 sphere_count = ray_tracer.scene->sphere_count;

    for (u32 i = 0; i < frame_buffer.size; i++)
        if (rayIntersectsWithSpheres(ray_tracer.closest_hit, ray_direction++, sphere, sphere_count))
            shadeClosestHitByNormal(ray_tracer.closest_hit, pixel++);
//            shadeRayByDirection(ray_direction++, pixel++);
        else
            (pixel++)->value = 0;
}

void generateRaysRT(Camera* camera) {
    generateRayDirections(
            ray_tracer.ray_directions,
            camera->focal_length,
            frame_buffer.width,
            frame_buffer.height);
}

void resizeRT(Controller* controller) {
    generateRaysRT(controller->camera);
}

void zoomRT(Controller* controller) {
    generateRaysRT(controller->camera);
    controller->changed.fov = false;
}

void rotateRT(Controller* controller) {
    transposeMatrix3D(controller->camera->transform->rotation, ray_tracer.inverted_camera_rotation);
    controller->changed.orientation = false;
    controller->changed.position = true;
}

void moveRT(Controller* controller) {
    Vector3* camera_position = controller->camera->transform->position;
    Sphere *sphere = ray_tracer.scene->spheres;
    u8 sphere_count = ray_tracer.scene->sphere_count;
    for (u8 i = 0; i < sphere_count; i++) {
        sub3D(sphere->world_position, camera_position, sphere->view_position);
        imul3D(sphere->view_position, ray_tracer.inverted_camera_rotation);
        sphere++;
    }

    controller->changed.position = false;
}

void initRayTracer(Engine* engine) {
    ray_tracer.renderer.title = "RayTrace";
    ray_tracer.renderer.render = rayTrace;
    ray_tracer.renderer.resize = resizeRT;
    ray_tracer.renderer.move = moveRT;
    ray_tracer.renderer.zoom = zoomRT;
    ray_tracer.renderer.rotate = rotateRT;
    ray_tracer.scene = &engine->scene;
    ray_tracer.rays_per_pixel = 1;
    ray_tracer.ray_count = frame_buffer.width * frame_buffer.height * ray_tracer.rays_per_pixel;
    ray_tracer.ray_directions = AllocN(Vector3, ray_tracer.ray_count);
    ray_tracer.closest_hit = Alloc(RayHit);
    initMatrix3x3(&ray_tracer.inverted_camera_rotation);
    setMatrix3x3ToIdentity(ray_tracer.inverted_camera_rotation);
}