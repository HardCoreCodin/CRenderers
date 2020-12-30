#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/camera.h"
#include "lib/globals/display.h"
#include "lib/shapes/line.h"

#define TAU 6.28f


typedef struct {
    vec3 position;
    f32 radius, thickness_radius;
    u32 revolution_count;
} Helix;

void drawHelix(Camera *camera, Helix *helix, Pixel *pixel) {
    u32 step_count = (u32)3600;
    f32 orbit_angle_step = TAU / (f32)step_count,
        helix_angle_step = orbit_angle_step * (f32)helix->revolution_count;

    vec3 current_position, previous_position;
    vec3 *center_position = &helix->position;

    vec3 center_to_orbit;
    center_to_orbit.x = helix->radius;
    center_to_orbit.y = center_to_orbit.z = 0;

    vec3 orbit_to_helix;
    orbit_to_helix.x = helix->thickness_radius;
    orbit_to_helix.y = orbit_to_helix.z = 0;

    vec3 orbit_to_helix_in_world_space;

    mat3 orbit_rotation;
    orbit_rotation.X.x = orbit_rotation.Z.z = cosf(orbit_angle_step);
    orbit_rotation.X.z = sinf(orbit_angle_step);
    orbit_rotation.Z.x = -orbit_rotation.X.z;
    orbit_rotation.X.y = orbit_rotation.Z.y = orbit_rotation.Y.x = orbit_rotation.Y.z =  0;
    orbit_rotation.Y.y = 1;

    mat3 helix_rotation;
    helix_rotation.X.x = helix_rotation.Y.y = cosf(helix_angle_step);
    helix_rotation.X.y = sinf(helix_angle_step);
    helix_rotation.Y.x = -helix_rotation.X.y;
    helix_rotation.X.z = helix_rotation.Y.z = helix_rotation.Z.x = helix_rotation.Z.y =  0;
    helix_rotation.Z.z = 1;

    vec3 projected_current_position, projected_previous_position;

    // Transform vertex positions of edges from view-space to screen-space (w/ culling and clipping):
    f32 x_factor = camera->focal_length;
    f32 y_factor = camera->focal_length * frame_buffer.dimentions.width_over_height;

    vec3 *cam_pos = &camera->transform.position;
    mat3 *cam_rot = &camera->transform.rotation_matrix_inverted;

    mat3 accumulated_orbit_rotation = orbit_rotation;

    for (u32 i = 0; i < step_count; i++) {
        imulVec3Mat3(&center_to_orbit, &orbit_rotation);
        imulVec3Mat3(&orbit_to_helix, &helix_rotation);
        mulVec3Mat3(&orbit_to_helix, &accumulated_orbit_rotation, &orbit_to_helix_in_world_space);

        addVec3(center_position, &center_to_orbit, &current_position);
        iaddVec3(&current_position, &orbit_to_helix_in_world_space);

        if (i) {
            // Project line-segment previous_position->current_position from world-space to screen-space, and draw it.

            subVec3(&current_position, cam_pos, &projected_current_position);
            subVec3(&previous_position, cam_pos, &projected_previous_position);

            imulVec3Mat3(&projected_current_position, cam_rot);
            imulVec3Mat3(&projected_previous_position, cam_rot);

            projectEdge(&projected_previous_position,
                        &projected_current_position,
                        x_factor,
                        y_factor);

            drawLine2D((i32)projected_previous_position.x,
                       (i32)projected_previous_position.y,
                       (i32)projected_current_position.x,
                       (i32)projected_current_position.y,
                       pixel);
        }

        imulMat3(&accumulated_orbit_rotation, &orbit_rotation);
        previous_position = current_position;
    }
}

typedef struct {
    vec3 position;
    f32 radius, height;
    u32 revolution_count;
} Coil;

void drawCoil(Camera *camera, Coil *coil, Pixel *pixel) {
    u32 step_count = (u32)3600;
    f32 angle_step = (TAU / (f32)step_count) * (f32)coil->revolution_count;
    f32 height_step = coil->height / (f32)step_count;

    vec3 current_position, previous_position;
    vec3 *center_position = &coil->position;

    vec3 center_to_coil;
    center_to_coil.x = coil->radius;
    center_to_coil.y = center_to_coil.z = 0;

    mat3 rotation;
    rotation.X.x = rotation.Z.z = cosf(angle_step);
    rotation.X.z = sinf(angle_step);
    rotation.Z.x = -rotation.X.z;
    rotation.X.y = rotation.Z.y = rotation.Y.x = rotation.Y.z =  0;
    rotation.Y.y = 1;

    vec3 projected_current_position, projected_previous_position;

    // Transform vertex positions of edges from view-space to screen-space (w/ culling and clipping):
    f32 x_factor = camera->focal_length;
    f32 y_factor = camera->focal_length * frame_buffer.dimentions.width_over_height;

    vec3 *cam_pos = &camera->transform.position;
    mat3 *cam_rot = &camera->transform.rotation_matrix_inverted;

    for (u32 i = 0; i < step_count; i++) {
        imulVec3Mat3(&center_to_coil, &rotation);
        addVec3(center_position, &center_to_coil, &current_position);

        if (i) {
            // Project line-segment previous_position->current_position from world-space to screen-space, and draw it.

            subVec3(&current_position, cam_pos, &projected_current_position);
            subVec3(&previous_position, cam_pos, &projected_previous_position);

            imulVec3Mat3(&projected_current_position, cam_rot);
            imulVec3Mat3(&projected_previous_position, cam_rot);

            projectEdge(&projected_previous_position,
                        &projected_current_position,
                        x_factor,
                        y_factor);

            drawLine2D((i32)projected_previous_position.x,
                       (i32)projected_previous_position.y,
                       (i32)projected_current_position.x,
                       (i32)projected_current_position.y,
                       pixel);
        }

        center_to_coil.y += height_step;
        previous_position = current_position;
    }
}


Coil my_coil;
Helix my_helix;
Pixel my_helix_pixel;