#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/nodes/scene.h"

typedef struct {
    f32 inner_hit_distance, outer_hit_distance;
    bool hit;
} SphereHit;
SphereHit sphere_hits[SPHERE_COUNT];

#define NORMAL_THRESHOLD 0.70710678118654746f

inline void getSphereHitPosition(vec3 *origin, vec3 *direction, f32 distance, vec3 *position) {
    scaleVec3(direction, distance, position);
    iaddVec3(position, origin);
}

inline void getSphereHitNormal(vec3 *position, vec3 *center, bool inner, f32 scale, vec3 *normal) {
    subVec3(inner ? center : position, inner ? position : center, normal);
    if (scale) iscaleVec3(normal, scale);
}

inline void getSphereHitUV(vec3 *normal, vec2 *uv) {
    if (normal->x > +NORMAL_THRESHOLD) { // right:
        uv->x = 0.5f + normal->z * 0.5f;
        uv->y = 0.5f + normal->y * 0.5f;
    } else if (normal->x < -NORMAL_THRESHOLD) { // left:
        uv->x = 0.5f - normal->z * 0.5f;
        uv->y = 0.5f + normal->y * 0.5f;
    } else if (normal->z > +NORMAL_THRESHOLD) { // front:
        uv->x = 0.5f + normal->x * 0.5f;
        uv->y = 0.5f + normal->y * 0.5f;
    } else if (normal->z < -NORMAL_THRESHOLD) { // back:
        uv->x = 0.5f - normal->x * 0.5f;
        uv->y = 0.5f + normal->y * 0.5f;
    } else if (normal->y > +NORMAL_THRESHOLD) { // top:
        uv->x = 0.5f + normal->x * 0.5f;
        uv->y = 0.5f + normal->z * 0.5f;
    } else if (normal->y < -NORMAL_THRESHOLD) { // bottom:
        uv->x = 0.5f + normal->x * 0.5f;
        uv->y = 0.5f - normal->z * 0.5f;
    }
}

bool hitSpheres(RayHit* closest_hit, Material** material_ptr, bool skip_out_of_view, Sphere *skip_sphere) {
    f32 t, dt, r, closest_distance = closest_hit->distance;
    vec3 _I, *I = &_I, _C, *C = &_C;
    bool found = false;

    f32 distance;
    vec3 *Rd = &closest_hit->ray_direction,
         *Ro = &closest_hit->ray_origin,
         *P = &closest_hit->position,
         *N = &closest_hit->normal;

    // Loop over all the spheres and intersect the ray against them:
    Sphere* hit_sphere = scene.spheres;
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
        if (skip_sphere == sphere ||
            (skip_out_of_view &&
             !sphere->in_view))
            continue;

        subVec3(&sphere->position, Ro, C);
        t = dotVec3(C, Rd);
        if (t > 0) {
            scaleVec3(Rd, t, I);
            isubVec3(I, C);
            r = sphere->radius;
            dt = r*r - squaredLengthVec3(I);
            if (dt > 0) { // Inside the sphere
                distance = t - sqrtf(dt);
                if (distance > 0 && distance < closest_distance) {
                    closest_distance = distance;
                    hit_sphere = sphere;
                    found = true;
                }
            }
        }
    }

    if (found) {
        closest_hit->distance = closest_distance;
        C = &hit_sphere->position;
        r = hit_sphere->radius;

        scaleVec3(Rd, closest_distance, P);
        iaddVec3(P, Ro);
        subVec3(P, C, N);
        if (r != 1) iscaleVec3(N, 1/r);

        *material_ptr = hit_sphere->material;
    }

    return found;
}

inline void populateHit(RayHit *hit, Sphere *sphere, f32 distance, bool inner, bool get_uv) {
    getSphereHitPosition(&hit->ray_origin, &hit->ray_direction, distance, &hit->position);
    getSphereHitNormal(&hit->position, &sphere->position, inner, sphere->radius == 1 ? 0 : 1.0f / sphere->radius, &hit->normal);
    if (get_uv) getSphereHitUV(&hit->normal, &hit->uv);

    hit->inner = inner;
    hit->distance = distance;
}
bool hitSpheres2(RayHit* closest_hit, Material** material_ptr, bool skip_out_of_view, Sphere *skip_sphere) {
    f32 t, dt, r, d, closest_distance = closest_hit->distance;
    vec3 _I, *I = &_I, _C, *C = &_C;
    bool found = false;
    u8 sphere_id;
    Sphere *sphere = scene.spheres;

    // Loop over all the spheres and intersect the ray against them:
    for (sphere_id = 0; sphere_id != SPHERE_COUNT; sphere_id++) {
        sphere_hits[sphere_id].hit = false;

        sphere = &scene.spheres[sphere_id];
        if (skip_sphere == sphere ||
           (skip_out_of_view &&
            !sphere->in_view))
            continue;

        subVec3(&sphere->position, &closest_hit->ray_origin, C);
        t = dotVec3(C, &closest_hit->ray_direction);
        if (t > 0) {
            scaleVec3(&closest_hit->ray_direction, t, I);
            isubVec3(I, C);
            r = sphere->radius;
            dt = r*r - squaredLengthVec3(I);
            if (dt > 0) { // Inside the sphere
                d = sqrtf(dt);
                sphere_hits[sphere_id].hit = true;
                sphere_hits[sphere_id].inner_hit_distance = d + t;
                sphere_hits[sphere_id].outer_hit_distance = d - t;
                found = true;
            }
        }
    }

    if (found) {
        bool found_inner = false, inner, outer, populate_hit = true;
        SphereHit *sphere_hit;
        for (sphere_id = 0; sphere_id != SPHERE_COUNT; sphere_id++) {
            sphere = &scene.spheres[sphere_id];
            sphere_hit = &sphere_hits[sphere_id];
            if (sphere_hit->hit) {
                inner = sphere_hit->inner_hit_distance < closest_distance;
                outer = sphere_hit->outer_hit_distance < closest_distance;
                if (inner || outer) {
                    populate_hit = true;

                    if (sphere->material->has_transparency) {
                        if (outer) {
                            populateHit(closest_hit, &scene.spheres[sphere_id], sphere_hit->outer_hit_distance, false, true);
                            if (closest_hit->uv.x > 0.5f &&
                                closest_hit->uv.y < 0.5f) {
                                if (inner) {
                                    populateHit(closest_hit, &scene.spheres[sphere_id], sphere_hit->inner_hit_distance, true, true);
                                    if (closest_hit->uv.x > 0.5f &&
                                        closest_hit->uv.y < 0.5f)
                                        continue;
                                }
                            }
                        } else {
                            populateHit(closest_hit, &scene.spheres[sphere_id], sphere_hit->inner_hit_distance, true, true);
                            if (closest_hit->uv.x > 0.5f &&
                                closest_hit->uv.y < 0.5f)
                                continue;
                        }

                        populate_hit = false;
                    }

                    found_inner = !outer;
                    sphere = &scene.spheres[sphere_id];
                    closest_distance = outer ? sphere_hit->outer_hit_distance : sphere_hit->inner_hit_distance;
                }
            }
        }

        if (populate_hit) populateHit(closest_hit, sphere, closest_distance, found_inner, false);

        *material_ptr = sphere->material;
    }

    return found;
}

bool hitPlanes(RayHit* closest_hit, Material** material_ptr) {
    vec3 *Rd = &closest_hit->ray_direction;
    vec3 *Ro = &closest_hit->ray_origin;
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
        subVec3(&plane->position, Ro, p);
        n = &plane->normal;

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
        closest_hit->normal = hit_plane->normal;
        closest_hit->distance = closest_hit_distance;
        closest_hit->inner = false;

        *material_ptr = hit_plane->material;
    }

    return found;
}

//bool hitSpheresDoubleSided(RayHit* closest_hit, RayHit* farther_hit) {
//    f32 distance1, distance2, t, dt, r, d2 = 100000, d1 = 100000;
//    vec3 *Rd = closest_hit->ray_direction,
//            *Ro = closest_hit->ray_origin,
//            *P1 = &closest_hit->position,
//            *N1 = &closest_hit->normal,
//            *P2 = &farther_hit->position,
//            *N2 = &farther_hit->normal;
//
//    vec3 _I, *I = &_I,
//            _C, *C = &_C;
//    bool found = false, found_double_sided = false, found_back_side = false;
//
//    // Loop over all the spheres and intersect the ray against them:
//    Sphere* hit_sphere = scene.spheres;
//    Sphere* last_sphere = scene.spheres + scene.sphere_count;
//    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
//        subVec3(sphere->position, Ro, C);
//        t = dotVec3(C, Rd);
//        scaleVec3(Rd, t, I);
//        isubVec3(I, C);
//        r = sphere->radius;
//        dt = r*r - squaredLengthVec3(I);
//        if (dt > 0) { // Inside the sphere
//            dt = sqrtf(dt);
//            distance1 = t - dt;
//            if (sphere->material_id == REFRACTION) {
//                distance2 = t + dt;
//                if (distance1 > 0 &&
//                    distance1 < d1) {
//                    d1 = distance1;
//                    d2 = distance2;
//                    hit_sphere = sphere;
//                    found = true;
//                    found_double_sided = true;
//                    found_back_side = false;
//                } else if (distance1 < 0 &&
//                           distance2 > 0 &&
//                           distance2 < d2) {
//                    d2 = distance2;
//                    hit_sphere = sphere;
//                    found = true;
//                    found_double_sided = true;
//                    found_back_side = true;
//                }
//            } else {
//                if (distance1 > 0 &&
//                    distance1 < d1) {
//                    d1 = distance1;
//                    hit_sphere = sphere;
//                    found = true;
//                    found_double_sided = false;
//                    found_back_side = false;
//                }
//            }
//        }
//    }
//
//    if (found) {
//        C = hit_sphere->position;
////        r = 1.0f / hit_sphere->radius;
//        closest_hit->material_id = hit_sphere->material_id;
//        closest_hit->distance = d1;
//        farther_hit->distance = d2;
//
//        if (found_double_sided) {
//            scaleVec3(Rd, d2, P2);
//            iaddVec3(P2, Ro);
//            subVec3(C, P2, N2);
////            iscaleVec3(N2, r);
//            if (found_back_side) closest_hit->distance = 0;
//        }
//
//        if (!found_double_sided || !found_back_side) {
//            scaleVec3(Rd, d1, P1);
//            iaddVec3(P1, Ro);
//            subVec3(P1, C, N1);
////            iscaleVec3(N1, r);
//        }
//    }
//
//    return found;
//}