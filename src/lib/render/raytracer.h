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

#include "lib/render/shaders/closest_hit/normal.h"
#include "lib/render/shaders/closest_hit/lambert.h"
#include "lib/render/shaders/closest_hit/phong.h"
#include "lib/render/shaders/closest_hit/blinn.h"
#include "lib/render/shaders/intersection/ray_sphere.h"
#include "lib/render/shaders/intersection/ray_plane.h"
#include "lib/render/shaders/generation/ray_generation.h"

static char* RAY_TRACER_TITLE = "RayTrace";

RayTracer ray_tracer;

typedef void (*Shader)(RayHit* closestHit, Pixel* pixel);
Shader current_shader = shadeClosestHitByNormal;
enum ShadingMode last_shading_mode = Normal;

inline bool inRange(range2i range, u16 value) {
    return value >= range.min &&
           value <= range.max;
}

inline bool inBounds(Bounds2Di *bounds, u16 x, u16 y) {
    return inRange(bounds->x_range, x) &&
           inRange(bounds->y_range, y);
}

inline bool pixelIsActive(u16 x, u16 y) {
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++)
        if (inBounds(&sphere->view_bounds, x, y)) {
            frame_buffer.active_pixel_count++;
            return true;
        }
    return false;
}

void onShadingModeChanged() {
    last_shading_mode = shading_mode;
    if      (shading_mode == Lambert) current_shader = shadeLambert;
    else if (shading_mode == Phong) current_shader = shadePhong;
    else if (shading_mode == Blinn) current_shader = shadeBlinn;
    else current_shader = shadeClosestHitByNormal;
    setShadingModeInHUD();
}

void onRender() {
    Pixel* pixel = frame_buffer.pixels;
    vec3* ray_direction = ray_tracer.ray_directions;

    frame_buffer.active_pixel_count = 0;
    for (u16 y = 0; y < frame_buffer.height; y++)
        for (u16 x = 0; x < frame_buffer.width; x++) {
            pixel->value = 0;
            rayIntersectsWithPlanes(&ray_tracer.closest_hit, ray_direction, scene.planes, scene.plane_count);
            if (pixelIsActive(x, y)) {
                if (ctrl_is_pressed) pixel->color = WHITE;
                else rayIntersectsWithSpheres(&ray_tracer.closest_hit, ray_direction, scene.spheres, scene.sphere_count);
            }
            current_shader(&ray_tracer.closest_hit, pixel);

            pixel++;
            ray_direction++;
        }
}

void generateRays() {
    generateRayDirections(
        ray_tracer.ray_directions,
        current_camera_controller->camera->focal_length,
        frame_buffer.width,
        frame_buffer.height
    );
}

void onZoom() {
    generateRays();
    current_camera_controller->zoomed = false;
}

void onTurn() {
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
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
        subVec3(sphere->world_position, camera_position, sphere->view_position);
        imulVec3Mat3(sphere->view_position, &ray_tracer.inverted_camera_rotation);

        // Check sphere's visibility:
        sphere->in_view = false;
        r = sphere->radius;
        x = sphere->view_position->x;
        y = sphere->view_position->y;
        z = sphere->view_position->z;

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

                    ff = f / z;
                    ff /= 100;
                    left -= ff;
                    right += ff;
                    bottom -= ff;
                    top += ff;
                    sphere->in_view = true;
                    sphere->view_bounds.x_range.min = (u16)(f * (w + max(-w, left)));
                    sphere->view_bounds.x_range.max = (u16)(f * (w + min(+w, right)));
                    sphere->view_bounds.y_range.max = frame_buffer.height - (u16)(f * (h + max(-h, bottom)));
                    sphere->view_bounds.y_range.min = frame_buffer.height - (u16)(f * (h + min(+h, top)));
                    scene.active_sphere_count++;
                }
            }
        }
    }

    Plane* last_plane = scene.planes + scene.plane_count;
    for (Plane* plane = scene.planes; plane != last_plane; plane++) {
        subVec3(plane->world_position, camera_position, plane->view_position);
        imulVec3Mat3(plane->view_position, &ray_tracer.inverted_camera_rotation);
        mulVec3Mat3(plane->world_normal, &ray_tracer.inverted_camera_rotation, plane->view_normal);
        plane->in_view = plane->view_normal->z < 0;
    }

    Light *last_light = scene.lights + scene.light_count;
    for (Light* light = scene.lights; light != last_light; light++) {
        subVec3(light->world_position, camera_position, light->view_position);
        imulVec3Mat3(light->view_position, &ray_tracer.inverted_camera_rotation);
    }

    current_camera_controller->moved = false;
}

void onResize() {
    generateRays();
    onMove();
}

void initRayTracer() {
    ray_tracer.rays_per_pixel = 1;
    ray_tracer.ray_count = ray_tracer.rays_per_pixel * frame_buffer.size;
    ray_tracer.ray_directions = AllocN(vec3, ray_tracer.ray_count);
    setMat3ToIdentity(&ray_tracer.inverted_camera_rotation);
    fillVec3(&ray_tracer.closest_hit.position, 0);
    fillVec3(&ray_tracer.closest_hit.normal, 0);
    ray_tracer.closest_hit.distance = 0;
}