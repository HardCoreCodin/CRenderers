#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math2D.h"
#include "lib/nodes/scene.h"

bool rayIntersectsWithSpheres(
        RayHit* closest_hit, // The hit structure of the closest intersection of the ray with the spheres
        Vector3* ray_direction  // The direction that the ray is aiming at
) {
    f32 r, r2, // The radius of the current sphere (and it's square)
    d, d2, // The distance from the origin to the position of the current intersection (and it's square)
    o2c, // The distance from the ray's origin to a position along the ray closest to the current sphere's center
    O2C, // The distance from the ray's origin to that position along the ray for the closest intersection
    r2_minus_d2, // The square of the distance from that position to the current intersection position
    R2_minus_D2, // The square of the distance from that position to the closest intersection position
    R = 0, // The radius of the closest intersecting sphere
    D = 100000; // The distance from the origin to the position of the closest intersection yet - squared

    Vector3 _t, _p;
    Vector3 *s, // The center position of the sphere of the current intersection
            *S, // The center position of the sphere of the closest intersection yet
            *p = &_p, // The position of the current intersection of the ray with the spheres
            *t = &_t;

    // Loop over all the spheres and intersect the ray against them:
    Sphere* sphere = scene.spheres;
    for (u8 i = 0; i < scene.sphere_count; i++) {
        s = sphere->view_position;
        r = sphere->radius;
        r2 = r*r;
        sphere++;

        o2c = dot3D(ray_direction, s);
        if (o2c > 0) {
            scale3D(ray_direction, o2c, p);
            sub3D(s, p, t);
            d2 = dot3D(t, t);
            if (d2 <= r2) {
                r2_minus_d2 = r2 - d2;
                d = o2c - r2_minus_d2;
                if (d > 0 && d <= D) {
                    S = s; D = d; R = r; O2C = o2c; R2_minus_D2 = r2_minus_d2;
                    closest_hit->position.x = p->x;
                    closest_hit->position.y = p->y;
                    closest_hit->position.z = p->z;
                }
            }
        }
    }

    if (R) {
        if (R2_minus_D2 > 0.001) {
            closest_hit->distance = O2C - sqrtf(R2_minus_D2);
            scale3D(ray_direction, closest_hit->distance, &closest_hit->position);
        }

        sub3D(&closest_hit->position, S, &closest_hit->normal);
        if (R != 1)
            iscale3D(&closest_hit->normal, 1 / R);

        return true;
    } else
        return false;
}