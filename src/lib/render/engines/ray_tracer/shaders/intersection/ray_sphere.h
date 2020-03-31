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

bool intersectRayWithSpheres(
        f32 ROx, f32 ROy, f32 ROz, // The position that the ray originates from
        f32 RDx, f32 RDy, f32 RDz, // The direction that the ray is aiming at
        Vector3* hit_position, // The position of the closest intersection of the ray with the spheres
        Vector3* hit_normal    // The direction of the surface of the sphere at the intersection position
) {
    f32 r, r2, // The radius of the current sphere (and it's square)
    d, d2, // The distance from the origin to the position of the current intersection (and it's square)
    o2c, // The distance from the ray's origin to a position along the ray closest to the current sphere's center
    O2C, // The distance from the ray's origin to that position along the ray for the closest intersection
    r2_minus_d2, // The square of the distance from that position to the current intersection position
    R2_minus_D2, // The square of the distance from that position to the closest intersection position
    sx, sy, sz, // The center position of the sphere of the current intersection
    Sx, Sy, Sz, // The center position of the sphere of the closest intersection yet
    px, py, pz, // The position of the closest intersection of the ray with the spheres yet
    R; // The radius of the closest intersecting sphere

    R = 0;
    f32 D = 100000; // The distance from the origin to the position of the closest intersection yet - squared

    // Loop over all the spheres and intersect the ray against them:
    Sphere* sphere = scene.spheres;
    for (u32 sphere_index = 0; sphere_index < scene.sphere_count; sphere_index++) {
        sx = sphere->position.x;
        sy = sphere->position.y;
        sz = sphere->position.z;
        r = sphere->radius;
        sphere++;

        r2 = r*r;

        o2c = (sx - ROx)*RDx + (sy - ROy)*RDy + (sz - ROz)*RDz;
        if (o2c > 0) {
            px = ROx + o2c*RDx;
            py = ROy + o2c*RDy;
            pz = ROz + o2c*RDz;

            d2 = (sx - px)*(sx - px) + (sy - py)*(sy - py) + (sz - pz)*(sz - pz);
            if (d2 <= r2) {
                r2_minus_d2 = r2 - d2;
                d = o2c - r2_minus_d2;
                if (d > 0 && d <= D) {
                    D = d; R = r; O2C = o2c; R2_minus_D2 = r2_minus_d2;
                    Sx = sx; hit_position->x = px;
                    Sy = sy; hit_position->y = py;
                    Sz = sz; hit_position->z = pz;
                }
            }
        }
    }

    if (R) {
        if (R2_minus_D2 > 0.001) {
            D = O2C - sqrtf(R2_minus_D2);
            hit_position->x = ROx + D*RDx;
            hit_position->y = ROy + D*RDy;
            hit_position->z = ROz + D*RDz;
        }

        if (R == 1) {
            hit_normal->x = hit_position->x - Sx;
            hit_normal->y = hit_position->y - Sy;
            hit_normal->z = hit_position->z - Sz;
        } else {
            f32 one_over_radius = 1 / R;
            hit_normal->x = (hit_position->x - Sx) * one_over_radius;
            hit_normal->y = (hit_position->y - Sy) * one_over_radius;
            hit_normal->z = (hit_position->z - Sz) * one_over_radius;
        }

        return true;
    } else
        return false;
}