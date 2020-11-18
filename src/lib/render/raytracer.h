#pragma once

#include "lib/core/types.h"
#include "lib/shapes/line.h"
#include "lib/input/keyboard.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/controllers/camera_controller.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"

#include "lib/render/shaders/closest_hit.h"
#include "lib/render/shaders/intersection.h"
#include "lib/render/shaders/ray_generation.h"

static char* RAY_TRACER_TITLE = "RayTrace";

RayTracer ray_tracer;

inline bool inRange(range2i range, u16 value) {
    return value >= range.min &&
           value <= range.max;
}

inline bool inBounds(Bounds2Di *bounds, u16 x, u16 y) {
    return inRange(bounds->x_range, x) &&
           inRange(bounds->y_range, y);
}

inline bool hasSpheres(u16 x, u16 y) {
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++)
        if (inBounds(&sphere->bounds, x, y)) {
            frame_buffer.active_pixel_count++;
            return true;
        }
    return false;
}

void onRender() {
    Pixel* pixel = frame_buffer.pixels;
    vec3* ray_direction = ray_tracer.ray_directions;
    vec3 color;

    ray_tracer.closest_hit.ray_origin = scene.camera->transform.position;
    Material* material;
    frame_buffer.active_pixel_count = 0;
    for (u16 y = 0; y < frame_buffer.height; y++)
        for (u16 x = 0; x < frame_buffer.width; x++) {
            if (ctrl_is_pressed) {
                pixel->value = 0;
                if (hasSpheres(x, y)) pixel->color = WHITE;
            } else {
                fillVec3(&color, 0);
                ray_tracer.closest_hit.hit_depth = 0;
                ray_tracer.closest_hit.distance = MAX_DISTANCE;
                ray_tracer.closest_hit.n2_over_n1 = IOR_GLASS;
                ray_tracer.closest_hit.n1_over_n2 = n1_over_n2_for_air_and_glass;
                ray_tracer.closest_hit.ray_direction = *ray_direction;
                hitPlanes(&ray_tracer.closest_hit, &material);
                hitCubes(&ray_tracer.closest_hit, &material);
//                if (hasSpheres(x, y)) hitSpheresSimple(&ray_tracer.closest_hit, &material, true, NULL);
                if (alt_is_pressed) shadeNormal(&ray_tracer.closest_hit.normal, ray_tracer.closest_hit.distance, &color);
                else shadeLambert(&ray_tracer.closest_hit, &color);

                pixel->color.R = color.x > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.x;
                pixel->color.G = color.y > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.y;
                pixel->color.B = color.z > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.z;
            }
            pixel++;
            ray_direction++;
        }
}

void generateRays() {
    generateRayDirections(
        ray_tracer.ray_directions,
        current_camera_controller->camera,
        frame_buffer.width,
        frame_buffer.height
    );
}

void onZoom() {
    generateRays();
    current_camera_controller->moved = true;
    current_camera_controller->zoomed = false;
}

void onTurn() {
    generateRays();
    transposeMat3(
            &current_camera_controller->camera->transform.rotation_matrix,
            &ray_tracer.inverted_camera_rotation);
    current_camera_controller->turned = false;
    current_camera_controller->moved = true;
}

void onMove() {
    vec3* camera_position = &current_camera_controller->camera->transform.position;
    const Sphere* last_sphere = scene.spheres + scene.sphere_count;
    scene.active_sphere_count = 0;
    f32 x, y, z, r, w, h, f, ff, left, right, top, bottom;
    vec3 position;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
        subVec3(&sphere->position, camera_position, &position);
        imulVec3Mat3(&position, &ray_tracer.inverted_camera_rotation);

        // Check sphere's visibility:
        sphere->in_view = false;
        r = sphere->radius;
        x = position.x;
        y = position.y;
        z = position.z;

        if (z > r) {
            // w / z = 1 / focal_length
            w = z * current_camera_controller->camera->one_over_focal_length;
            left = x - r;
            right = x + r;
            if ((x > 0 && left < +w) ||
                (x < 0 && right > -w)) {
                // h / w = frame_buffer.height / frame_buffer.width
                h = w * frame_buffer.height_over_width;
                top = y + r;
                bottom = y - r;
                if ((y > 0 && bottom < +h) ||
                    (y < 0 && top > -h)) {
                    f = (f32)frame_buffer.width / (w + w);

                    ff = f / (z * r/2 * current_camera_controller->camera->focal_length);
                    left -= ff;
                    right += ff;
                    bottom -= ff;
                    top += ff;
                    sphere->in_view = true;
                    sphere->bounds.x_range.min = (u16)(f * (w + max(-w, left)));
                    sphere->bounds.x_range.max = (u16)(f * (w + min(+w, right)));
                    sphere->bounds.y_range.max = frame_buffer.height - (u16)(f * (h + max(-h, bottom)));
                    sphere->bounds.y_range.min = frame_buffer.height - (u16)(f * (h + min(+h, top)));
                    scene.active_sphere_count++;
                }
            }
        }
    }

    current_camera_controller->moved = false;
}

void onResize() {
    generateRays();
    onMove();
}

vec3 origin;

void initRayTracer() {
    ray_tracer.rays_per_pixel = 1;
    ray_tracer.ray_count = ray_tracer.rays_per_pixel * frame_buffer.size;
    ray_tracer.ray_directions = AllocN(vec3, ray_tracer.ray_count);
    setMat3ToIdentity(&ray_tracer.inverted_camera_rotation);
    fillVec3(&ray_tracer.closest_hit.position, 0);
    fillVec3(&ray_tracer.closest_hit.normal, 0);
    ray_tracer.closest_hit.distance = 0;
    ray_tracer.closest_hit.ray_origin = scene.camera->transform.position;
    fillVec3(&origin, 0);
    initShaders();
}