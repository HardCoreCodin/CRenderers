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

inline u8 getSphereVisibilityMask(Sphere *spheres, u8 sphere_count, u16 x, u16 y) {
    u8 mask = 0;

    for (u8 i = 0; i < sphere_count; i++) mask |= ((1 << i) * (
            spheres[i].in_view &&
            x >= spheres[i].bounds.x_range.min &&
            x <= spheres[i].bounds.x_range.max &&
            y >= spheres[i].bounds.y_range.min &&
            y <= spheres[i].bounds.y_range.max));

    return mask;
}

inline u8 renderPixel(
        Pixel* pixel,

        u16 x,
        u16 y,

        vec3* ray_origin,
        vec3 *ray_direction,

        Material *materials,
        Sphere* spheres,
        u8 sphere_count,

        bool shade_normals,
        bool shade_visibility) {
    u8 sphere_visibility_mask = getSphereVisibilityMask(spheres, sphere_count, x, y);
    if (shade_visibility) {
        pixel->color = sphere_visibility_mask ? WHITE : BLACK;
        return sphere_visibility_mask;
    }

    RayHit hit = {
            .ray_origin = *ray_origin,
            .ray_direction = *ray_direction,
            .hit_depth = 0,
            .distance = MAX_DISTANCE,
            .n1_over_n2 = n1_over_n2_for_air_and_glass,
            .n2_over_n1 = IOR_GLASS
    };
    vec3 color = {
            .x = 0,
            .y = 0,
            .z = 0
    };
    hitPlanes(&hit);
//                hitCubes(&hit);

//                perfStart(&aux_timer);
//                if (alt_is_pressed)
//                    hitImplicitTetrahedra(&hit);
//                else
//                    hitTetrahedra(&hit);
//                perfEnd(&aux_timer, aux_timer.accumulated_ticks >= ticks_per_second, i == frame_buffer.size);
    if (sphere_visibility_mask) {
        frame_buffer.active_pixel_count++;
        hitSpheresSimple(&hit,true, sphere_visibility_mask);
    }
    if (shade_normals)
        shadeNormal(&hit.normal, hit.distance, &color);
    else
        shadeLambert(&hit, &color);
//                if (materials[hit.material_id].uses & PHONG)
//                    shadePhong(&hit, &color);
//                else
//                    shadeLambert(&hit, &color);

    pixel->color.R = color.x > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.x;
    pixel->color.G = color.y > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.y;
    pixel->color.B = color.z > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.z;

    return sphere_visibility_mask;
}

void onRender() {
    Pixel* pixel = frame_buffer.pixels;
    vec3 *Rd = ray_tracer.ray_directions;
    vec3 *Ro = &scene.camera->transform.position;

    frame_buffer.active_pixel_count = 0;
    for (u16 y = 0; y < frame_buffer.height; y++)
        for (u16 x = 0; x < frame_buffer.width; x++, pixel++, Rd++)
            if (renderPixel(pixel,
                            x, y,

                            Ro, Rd,

                            scene.materials,
                            scene.spheres,
                            scene.sphere_count,

                            ctrl_is_pressed,
                            alt_is_pressed))
                frame_buffer.active_pixel_count++;
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
    ray_tracer.ray_count = ray_tracer.rays_per_pixel * MAX_WIDTH * MAX_HEIGHT;
    ray_tracer.ray_directions = AllocN(vec3, ray_tracer.ray_count);
    setMat3ToIdentity(&ray_tracer.inverted_camera_rotation);
    fillVec3(&origin, 0);
    initShaders();

//    ray_tracer.pixel_shapes_mask = AllocN(u8, ray_tracer.ray_count);
//    INV_SPHERE_MASK = 0;
//    for (u8 i = scene.sphere_count; i < 8; i++) INV_SPHERE_MASK |= 1 << i;
}