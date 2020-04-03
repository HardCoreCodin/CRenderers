#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math2D.h"
#include "lib/nodes/scene.h"

bool rayIntersectsWithSpheres(
        Vector3* RO, // The position that the ray originates from
        Vector3* RD, // The direction that the ray is aiming at
        Vector3* hit_position, // The position of the closest intersection of the ray with the spheres
        Vector3* hit_normal    // The direction of the surface of the sphere at the intersection position
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
        s = &sphere->position;
        r = sphere->radius;
        r2 = r*r;
        sphere++;

        sub3D(s, RO, t);
        o2c = dot3D(t, RD);
        if (o2c > 0) {
            scale3D(RD, o2c, t);
            add3D(RO, t, p);
            sub3D(s, p, t);
            d2 = dot3D(t, t);
            if (d2 <= r2) {
                r2_minus_d2 = r2 - d2;
                d = o2c - r2_minus_d2;
                if (d > 0 && d <= D) {
                    S = s; D = d; R = r; O2C = o2c; R2_minus_D2 = r2_minus_d2;
                    hit_position->x = p->x;
                    hit_position->y = p->y;
                    hit_position->z = p->z;
                }
            }
        }
    }

    if (R) {
        if (R2_minus_D2 > 0.001) {
            scale3D(RD, O2C - sqrtf(R2_minus_D2), t);
            add3D(RO, t, hit_position);
        }

        sub3D(hit_position, S, hit_normal);
        if (R != 1)
            iscale3D(hit_normal, 1 / R);

        return true;
    } else
        return false;
}