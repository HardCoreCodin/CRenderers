#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math2D.h"
#include "lib/nodes/scene.h"

bool rayIntersectsWithPlanes(
        RayHit* closest_hit, // The hit structure of the closest intersection of the ray with the spheres
        vec3* ray_direction,  // The direction that the ray is aiming at
        Plane* planes,
        u8 plane_count) {

    vec3 *closest_normal = &closest_hit->normal;
    vec3 *N, *P;
    f32 Rd_dot_N,
        P_dot_N,
        T = 10000,
        t = 0;

    bool hit_found = false;

    // Loop over all the planes and intersect the ray against them:
    Plane* last_plane = planes + plane_count;
    for (Plane* plane = planes; plane != last_plane; plane++) {
//        if (!plane->in_view) continue;

        N = plane->view_normal;
        P = plane->view_position;

        Rd_dot_N = dotVec3(ray_direction, N);
        if (Rd_dot_N >= 0 ||
           -Rd_dot_N < EPS)
            continue;

        P_dot_N = dotVec3(P, N);
        if (P_dot_N >= 0 ||
           -P_dot_N < EPS)
            continue;

        t = P_dot_N / Rd_dot_N;
        if (t < T) {
            T = t;
            closest_normal = N;
            hit_found = true;
        }
    }

    if (hit_found) {
        scaleVec3(ray_direction, T, &closest_hit->position);
        closest_hit->normal = *closest_normal;
        closest_hit->distance = T;
        closest_hit->ray_direction = ray_direction;
    }

    return hit_found;
}