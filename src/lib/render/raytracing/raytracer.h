#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/nodes/camera.h"
#include "lib/input/controllers.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"
#include "lib/render/raytracing/raytrace_types.h"
#include "lib/render/raytracing/shaders/closest_hit/normal.h"
#include "lib/render/raytracing/shaders/intersection/ray_sphere.h"
#include "lib/render/raytracing/shaders/generation/ray_generation.h"

char* TITLE = "RayTrace";

RayTracer renderer;
RayHit* closest_hit;

inline void render() {
    Pixel* pixel = (Pixel*)frame_buffer.pixels;
    Vector3* ray_direction = renderer.ray_directions;

    for (u32 i = 0; i < frame_buffer.size; i++)
        if (rayIntersectsWithSpheres(closest_hit, ray_direction++))
            shadeClosestHitByNormal(closest_hit, pixel++);
//            shadeRayByDirection(ray_direction++, pixel++);
        else
            (pixel++)->value = 0;
}

inline void generateRays() {
    generateRayDirections(
            renderer.ray_directions,
            renderer.camera.focal_length,
            frame_buffer.width,
            frame_buffer.height);
}

inline void onResized() {generateRays();}
inline void onZoomed() {generateRays();}
inline void onMoved() {
    Vector3* camera_position = renderer.camera.transform->position;
    Matrix3x3* inverted_camera_rotation = renderer.camera.transform->rotation_inverted;
    transposeMatrix3D(renderer.camera.transform->rotation, inverted_camera_rotation);

    Sphere *sphere = scene.spheres;
    for (u8 i = 0; i < scene.sphere_count; i++) {
        sub3D(sphere->world_position, camera_position, sphere->view_position);
        imul3D(sphere->view_position, inverted_camera_rotation);
        sphere++;
    }
}


void initRenderer() {
    renderer.rays_per_pixel = 1;
    renderer.ray_count = frame_buffer.width * frame_buffer.height * renderer.rays_per_pixel;
    renderer.ray_directions = (Vector3*)allocate(sizeof(Vector3) * renderer.ray_count);

    initCamera3D(&renderer.camera);
    renderer.camera.transform->position->x = 5;
    renderer.camera.transform->position->y = 5;
    renderer.camera.transform->position->z = -10;
    onMoved();

    closest_hit = (RayHit*)allocate(sizeof(RayHit));
}