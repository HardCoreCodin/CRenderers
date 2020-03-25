#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/core/memory.h"
#include "lib/math/math2D.h"
#include "lib/render/engines/ray_tracer/ray.h"
#include "lib/render/engines/ray_tracer/scene.h"

Vector3* ray_origin_to_sphere_center;
Vector3* ray_origin_to_closest_position;
Vector3* closest_position_to_sphere_center;

u8 rayIntersectSpheres(Ray* ray) {
    f32 squared_radius = 0;
    f32 squared_distance = 0;
    
    RayHit* closest_hit = ray->closest_hit;
    RayHit* current_hit = ray->current_hit;

    f32* current_distance = &current_hit->origin_to_closest_minus_delta_of_squares;
    f32* closest_distance = &closest_hit->origin_to_closest_minus_delta_of_squares;
    *closest_distance = 10000;

    Sphere* current_sphere = scene.spheres;
    Sphere* closest_sphere = 0;

    for (u32 sphere_index = 0; sphere_index < scene.sphere_count; sphere_index++) {
        squared_radius = current_sphere->radius * current_sphere->radius;

        sub3D(&current_sphere->position, ray->origin, ray_origin_to_sphere_center);

        current_hit->origin_to_closest = dot3D(ray_origin_to_sphere_center, ray->direction);
        if (current_hit->origin_to_closest > 0) {

            scale3D(ray->direction, current_hit->origin_to_closest, ray_origin_to_closest_position);
            add3D(ray->origin, ray_origin_to_closest_position, &current_hit->position);
            sub3D(&current_sphere->position, &current_hit->position, closest_position_to_sphere_center);

            squared_distance = squaredLength3D(closest_position_to_sphere_center);
            if (squared_distance <= squared_radius) {

                current_hit->delta_of_squares = squared_radius - squared_distance;
                *current_distance = current_hit->origin_to_closest - current_hit->delta_of_squares;
                if (*current_distance > 0 &&
                    *current_distance <= *closest_distance) {
                    *closest_hit = *current_hit;
                    closest_sphere = current_sphere;
                }
            }
        }

        current_sphere++;
    }

    if (closest_sphere) {
        if (closest_hit->delta_of_squares > 0.001f) {
            closest_hit->distance = closest_hit->origin_to_closest - sqrtf(closest_hit->delta_of_squares);
            scale3D(ray->direction, closest_hit->distance, &closest_hit->position);
            iadd3D(&closest_hit->position, ray->origin);
        }

        sub3D(&closest_hit->position, &closest_sphere->position, &closest_hit->normal);
        if (closest_sphere->radius != 1)
            iscale3D(&closest_hit->normal, 1 / closest_sphere->radius);
    }

    return closest_sphere ? true : false;
}

void initRaySphereIntersection() {
    ray_origin_to_sphere_center = (Vector3*)allocate(sizeof(Vector3));
    ray_origin_to_closest_position = (Vector3*)allocate(sizeof(Vector3));
    closest_position_to_sphere_center = (Vector3*)allocate(sizeof(Vector3));
}