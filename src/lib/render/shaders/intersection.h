#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/nodes/scene.h"

bool hitSpheres(RayHit* closest_hit, bool skip) {
    f32 distance, t, dt, r, d = 100000;
    vec3 *Rd = closest_hit->ray_direction,
            *Ro = closest_hit->ray_origin,
            *P = &closest_hit->position,
            *N = &closest_hit->normal;

    vec3 _I, *I = &_I,
            _C, *C = &_C;
    bool found = false;

    // Loop over all the spheres and intersect the ray against them:
    Sphere* hit_sphere = scene.spheres;
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
        if (skip && !sphere->in_view) continue;
        subVec3(sphere->position, Ro, C);
        t = dotVec3(C, Rd);
        if (t > 0) {
            scaleVec3(Rd, t, I);
            isubVec3(I, C);
            r = sphere->radius;
            dt = r*r - squaredLengthVec3(I);
            if (dt > 0) { // Inside the sphere
                distance = t - sqrtf(dt);
                if (distance > 0 && distance < d) {
                    d = distance;
                    hit_sphere = sphere;
                    found = true;
                }
            }
        }
    }

    if (found) {
        closest_hit->distance = d;
        C = hit_sphere->position;
        r = hit_sphere->radius;

        scaleVec3(Rd, d, P);
        iaddVec3(P, Ro);
        subVec3(P, C, N);
        if (r != 1) iscaleVec3(N, 1/r);

        closest_hit->material_id = hit_sphere->material_id;
    }

    return found;
}

bool hitSpheresDoubleSided(RayHit* closest_hit, RayHit* farther_hit) {
    f32 distance1, distance2, t, dt, r, d2 = 100000, d1 = 100000;
    vec3 *Rd = closest_hit->ray_direction,
         *Ro = closest_hit->ray_origin,
         *P1 = &closest_hit->position,
         *N1 = &closest_hit->normal,
         *P2 = &farther_hit->position,
         *N2 = &farther_hit->normal;

    vec3 _I, *I = &_I,
         _C, *C = &_C;
    bool found = false, found_double = false;

    // Loop over all the spheres and intersect the ray against them:
    Sphere* hit_sphere = scene.spheres;
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
        subVec3(sphere->position, Ro, C);
        t = dotVec3(C, Rd);
        scaleVec3(Rd, t, I);
        isubVec3(I, C);
        r = sphere->radius;
        dt = r*r - squaredLengthVec3(I);
        if (dt > 0) { // Inside the sphere
            dt = sqrtf(dt);
            distance1 = t - dt;
            distance2 = t + dt;
            if (distance1 > 0 &&
                distance1 < d1) {
                d1 = distance1;
                d2 = distance2;
                hit_sphere = sphere;
                found = true;
                found_double = true;
            } else if (distance1 < 0 &&
                       distance2 > 0 &&
                       distance2 < d2) {
                d2 = distance2;
                hit_sphere = sphere;
                found = true;
                found_double = false;
            }
        }
    }

    if (found) {
        farther_hit->distance = d2;
        C = hit_sphere->position;
        r = hit_sphere->radius == 1 ? 1 : 1/hit_sphere->radius;

        scaleVec3(Rd, d2, P2);
        iaddVec3(P2, Ro);
        subVec3(C, P2, N2);
        if (r != 1) iscaleVec3(N2, r);

        if (found_double) {
            closest_hit->distance = d1;
            scaleVec3(Rd, d1, P1);
            iaddVec3(P1, Ro);
            subVec3(P1, C, N1);
            if (r != 1) iscaleVec3(N1, r);
        } else closest_hit->distance = 0;

        closest_hit->material_id = hit_sphere->material_id;
        farther_hit->material_id = hit_sphere->material_id;
    }

    return found;
}

bool hitPlanes(RayHit* closest_hit) {
    vec3 *Rd = closest_hit->ray_direction;
    vec3 *Ro = closest_hit->ray_origin;
    vec3 *n, _p, *p = &_p;
    f32 Rd_dot_n,
        p_dot_n,
        closest_hit_distance = 10000,
        hit_distance = 0;

    bool found = false;

    // Loop over all the planes and intersect the ray against them:
    Plane* hit_plane = scene.planes;
    Plane* last_plane = scene.planes + scene.plane_count;
    for (Plane* plane = scene.planes; plane != last_plane; plane++) {
        subVec3(plane->position, Ro, p);
        n = plane->normal;

        Rd_dot_n = dotVec3(Rd, n);
        if (Rd_dot_n >= 0 ||
            -Rd_dot_n < EPS)
            continue;

        p_dot_n = dotVec3(p, n);
        if (p_dot_n >= 0 ||
            -p_dot_n < EPS)
            continue;

        hit_distance = p_dot_n / Rd_dot_n;
        if (hit_distance < closest_hit_distance) {
            closest_hit_distance = hit_distance;
            hit_plane = plane;
            found = true;
        }
    }

    if (found) {
        scaleVec3(Rd, closest_hit_distance, &closest_hit->position);
        iaddVec3(&closest_hit->position, Ro);
        closest_hit->normal = *hit_plane->normal;
        closest_hit->distance = closest_hit_distance;
        closest_hit->material_id = hit_plane->material_id;
    }

    return found;
}