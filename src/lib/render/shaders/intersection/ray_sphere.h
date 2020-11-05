#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math2D.h"
#include "lib/nodes/scene.h"

bool rayIntersectsWithSpheres(
        RayHit* closest_hit, // The hit structure of the closest intersection of the ray with the spheres
        vec3* ray_direction,  // The direction that the ray is aiming at
        Sphere* spheres,
        u8 sphere_count) {
    f32 r, r2, // The radius of the current sphere (and it's square)
    d, d2, // The distance from the origin to the position of the current intersection (and it's square)
    o2c = 0, // The distance from the ray's origin to a position along the ray closest to the current sphere's center
    O2C = 0, // The distance from the ray's origin to that position along the ray for the closest intersection
    r2_minus_d2 = 0, // The square of the distance from that position to the current intersection position
    R2_minus_D2 = 0, // The square of the distance from that position to the closest intersection position
    R = 0, // The radius of the closest intersecting sphere
    D = 100000; // The distance from the origin to the position of the closest intersection yet - squared

    vec3 _t, _p;
    vec3 *s, // The center position of the sphere of the current intersection
         *S, // The center position of the sphere of the closest intersection yet
         *p = &_p, // The position of the current intersection of the ray with the spheres
         *t = &_t;
    S = spheres->view_position;

    // Loop over all the spheres and intersect the ray against them:
    Sphere* last_sphere = spheres + sphere_count;
    for (Sphere* sphere = spheres; sphere != last_sphere; sphere++) {
        if (!sphere->in_view) continue;

        s = sphere->view_position;
        r = sphere->radius;
        r2 = r*r;

        o2c = dotVec3(ray_direction, s);
        if (o2c > 0) {
            scaleVec3(ray_direction, o2c, p);
            subVec3(s, p, t);
            d2 = squaredLengthVec3(t);
            if (d2 <= r2) {
                r2_minus_d2 = r2 - d2;
                d = o2c - r2_minus_d2;
                if (d > 0 && d < D) {
                    S = s;
                    D = d;
                    R = r;
                    O2C = o2c;
                    R2_minus_D2 = r2_minus_d2;
                }
            }
        }
    }

    if (R) {
        if (R2_minus_D2 > 0) {
            closest_hit->distance = O2C - sqrtf(R2_minus_D2);
            scaleVec3(ray_direction, closest_hit->distance, &closest_hit->position);
        }

        subVec3(&closest_hit->position, S, &closest_hit->normal);
        if (R != 1)
            iscaleVec3(&closest_hit->normal, 1 / R);

        closest_hit->ray_direction = ray_direction;

        return true;
    } else
        return false;
}