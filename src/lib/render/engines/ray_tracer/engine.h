#pragma once

#include "lib/core/types.h"
#include "lib/core/memory.h"

#include "lib/controls/fps.h"
#include "lib/controls/orbit.h"
#include "lib/controls/controls.h"

#include "lib/nodes/camera.h"

#include "lib/render/buffers.h"
#include "lib/render/engines/ray_tracer/ray.h"
#include "lib/render/engines/ray_tracer/engine.h"
#include "lib/render/engines/ray_tracer/shaders/closest_hit/normal.h"
#include "lib/render/engines/ray_tracer/shaders/ray_generation/ray_generator.h"
#include "lib/render/engines/ray_tracer/shaders/intersection/ray_sphere.h"

static char* TITLE = "RayTrace";

typedef struct RenderEngine {
    Camera3D camera;

    Ray ray;
    u32 ray_count;
    u8 rays_per_pixel;

    Vector3* source_ray_directions;
    Vector3* ray_directions;
} RenderEngine;

static RenderEngine engine;

void initRenderEngine() {
    engine.rays_per_pixel = 1;
    engine.ray_count = frame_buffer.width * frame_buffer.height * engine.rays_per_pixel;

    engine.source_ray_directions = (Vector3*)allocate(sizeof(Vector3) * engine.ray_count);
    engine.ray_directions = (Vector3*)allocate(sizeof(Vector3) * engine.ray_count);

    initCamera3D(&engine.camera);
    engine.camera.transform->position->x = 5;
    engine.camera.transform->position->y = 5;
    engine.camera.transform->position->z = -10;

    initRay(&engine.ray);
    engine.ray.origin = engine.camera.transform->position;
    engine.ray.direction = engine.ray_directions;

    initScene();
    initRaySphereIntersection();
}

void render() {
    engine.ray.direction = engine.ray_directions;
    u32* pixel = frame_buffer.pixels;

    for (u32 i = 0; i < frame_buffer.size; i++) {
        if (rayIntersectSpheres(&engine.ray))
            shadeByNormal(pixel++, engine.ray.closest_hit);
        else
            *pixel++ = 0;

        engine.ray.direction++;
    }
}

void rotateRayDirections() {
    Matrix3x3* rotation_matrix = engine.camera.transform->rotation;
    Vector3* source_ray_direction = engine.source_ray_directions;
    Vector3* ray_direction = engine.ray_directions;

    for (u32 ray_index = 0; ray_index < engine.ray_count; ray_index++)
        mul3D(source_ray_direction++, rotation_matrix, ray_direction++);
}

void onFrameBufferResized() {
    generateRayDirections(
            engine.source_ray_directions,
            engine.camera.focal_length,
            frame_buffer.width,
            frame_buffer.height);
    rotateRayDirections();
}

void onMousePositionChanged(f32 dx, f32 dy) {
    if (controls.mouse.is_captured)
        onOrient(-dx, -dy);
    else if (controls.mouse.pressed) {
        if (controls.mouse.pressed & controls.buttons.MIDDLE)
            onPan(-dx, dy);
        else if (controls.mouse.pressed & controls.buttons.RIGHT)
            onOrbit(-dx, -dy);
    }
}

void onMouseWheelChanged(f32 amount) {
    if (controls.mouse.is_captured)
        onZoom(amount);
    else
        onDolly(amount);
}

void update(f32 delta_time) {
    if (delta_time > 1)
        delta_time = 1;

    if (controls.mouse.is_captured) {

        if (zoom.changed) {
            updateZoom(&engine.camera.focal_length);
            generateRayDirections(
                    engine.source_ray_directions,
                    engine.camera.focal_length,
                    frame_buffer.width,
                    frame_buffer.height);
            rotateRayDirections();
        }

        if (orientation.changed) {
            updateOrientation(engine.camera.transform);
            rotateRayDirections();
        }
    } else {
        if (orbit.changed) {
            updateOrbit(engine.camera.transform);
            rotateRayDirections();
        }

        if (pan.changed)
            updatePan(engine.camera.transform);

        if (dolly.changed)
            updateDolly(engine.camera.transform);
    }

    updatePosition(engine.camera.transform, delta_time);
}