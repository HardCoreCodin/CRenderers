#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"
#include "lib/globals/raytracing.h"
#include "common.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitPlane(vec3 *position, vec3 *normal, vec3 *ray_direction, vec3 *ray_origin, f32 *hit_distance) {
    f32 Rd_dot_n = dotVec3(ray_direction, normal);
    if (Rd_dot_n >= 0 ||
        -Rd_dot_n < EPS)
        return false;

    vec3 ray_origin_to_position;
    subVec3(position, ray_origin, &ray_origin_to_position);
    f32 p_dot_n = dotVec3(&ray_origin_to_position, normal);
    if (p_dot_n >= 0 ||
        -p_dot_n < EPS)
        return false;

    *hit_distance = p_dot_n / Rd_dot_n;
    return true;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitPlanes(Plane *planes, Ray* ray) {
    vec3 *Ro = ray->origin,
         *Rd = ray->direction;
    bool found = false;
    f32 current_hit_distance, closest_hit_distance = ray->hit.distance;

    // Loop over all the planes and intersect the ray against them:
    Plane* hit_plane = planes;
    Plane* plane;
    for (u8 i = 0; i < PLANE_COUNT; i++) {
        plane = &planes[i];
        if (!hitPlane(&plane->position, &plane->normal, Rd, Ro, &current_hit_distance))
            continue;

        if (current_hit_distance < closest_hit_distance) {
            closest_hit_distance = current_hit_distance;
            hit_plane = plane;
            found = true;
        }
    }

    if (found) {
        setRayHitPosition(Ro, Rd, closest_hit_distance - EPS, &ray->hit.position);
        ray->hit.material_id = hit_plane->material_id;
        ray->hit.normal = hit_plane->normal;
        ray->hit.distance = closest_hit_distance;
        f32 x = ray->hit.position.x + 32; x *= 0.125f; x -= (f32)((u8)x);
        f32 y = ray->hit.position.y + 32; y *= 0.125f; y -= (f32)((u8)y);
        f32 z = ray->hit.position.z + 32; z *= 0.125f; z -= (f32)((u8)z);
        if (       hit_plane->normal.x > 0) { // Left plane (facing right):
            ray->hit.uv.x = z;
            ray->hit.uv.y = y;
        } else if (hit_plane->normal.x < 0) { // Right plane (facing left):
            ray->hit.uv.x = 1 - z;
            ray->hit.uv.y = y;
        } else if (hit_plane->normal.y > 0) { // Bottom plane (facing up):
            ray->hit.uv.x = x;
            ray->hit.uv.y = z;
        } else if (hit_plane->normal.y < 0) { // Top plane (facing down):
            ray->hit.uv.x = x;
            ray->hit.uv.y = 1 - z;
        } else if (hit_plane->normal.z > 0) { // Back plane (facing forward):
            ray->hit.uv.x = 1 - x;
            ray->hit.uv.y = y;
        } else if (hit_plane->normal.z < 0) { // Front plane (facing backward):
            ray->hit.uv.x = x;
            ray->hit.uv.y = y;
        }
        ray->hit.is_back_facing = false;
    }

    return found;
}