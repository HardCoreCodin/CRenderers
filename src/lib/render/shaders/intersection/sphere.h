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
vec2 setRaySphereHit(vec3 *Ro, vec3 *Rd, vec3 *P, vec3 *N, vec3 *C, mat3 *M, f32 d, bool invert_normal) {
    vec3 n;
    setRayHitPosition(Ro, Rd, d, P);
    setRayHitDirection(P, C, N);
    mulVec3Mat3(N, M, &n);
    if (invert_normal) invertVec3(N);
    return getUV(&n);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitSpheresSimple(Sphere *spheres, Ray *ray) {
    vec3 *Ro = ray->origin,
         *Rd = ray->direction,
         *P  = &ray->hit.position,
         *N  = &ray->hit.normal;
    f32 t, dt, r, closest_distance = ray->hit.distance;
    vec3 _i, *I = &_i, _c, *C = &_c;
    bool found = false;

    u8 visibility_mask = ray->masks.visibility.spheres;
    f32 distance;
    u8 sphere_id = 1;
    Sphere *sphere;
    Sphere *hit_sphere;

    // Loop over all the spheres and intersect the ray against them:
    for (u8 i = 0; i < SPHERE_COUNT; i++, sphere_id <<= (u8)1) {
        if (!(sphere_id & visibility_mask)) continue;

        sphere = &spheres[i];

        subVec3(&sphere->position, Ro, C);
        t = dotVec3(C, Rd);
        if (t > 0) {
            scaleVec3(Rd, t, I);
            isubVec3(I, C);
            r = sphere->radius;
            dt = r*r - squaredLengthVec3(I);
            if (dt > 0 && t*t > dt) { // Inside the sphere
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
        ray->hit.is_back_facing = false;
        ray->hit.material_id = hit_sphere->material_id;
        ray->hit.distance = closest_distance;
        C = &hit_sphere->position;
        r = hit_sphere->radius;

        scaleVec3(Rd, closest_distance, P);
        iaddVec3(P, Ro);
        subVec3(P, C, N);
        if (r != 1) iscaleVec3(N, 1/r);
    }

    return found;
}

#define isTransparent(uv) (((u8)(uv.x * 4) % 2) ? (((u8)((uv.y + 0.25) * 4)) % 2) : (((u8)(uv.y * 4)) % 2))


#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitSpheres(Sphere *spheres, Ray *ray, bool check_any) {
    RayHit current_hit, closest_hit = ray->hit;
    vec3 *P = &current_hit.position,
         *N = &current_hit.normal,
         *Ro = ray->origin,
         *Rd = ray->direction;

    u8 transparency_mask = ray->masks.transparency.spheres;
    u8 visibility_mask   = ray->masks.visibility.spheres;

    f32 t, dt, r, d, closest_distance_squared = ray->hit.distance * ray->hit.distance;
    vec3 _i, *I = &_i, _c, *C = &_c;
    vec3 *Sp;
    mat3 *Sr;
    bool has_inner_hit,
         has_outer_hit,
         found = false;

    u8 sphere_id = 1;
    Sphere *sphere = spheres;

    f32 outer_hit_distance,
        inner_hit_distance;

    // Loop over all the spheres and intersect the ray against them:
    for (u8 i = 0; i < SPHERE_COUNT; i++, sphere_id <<= (u8)1, sphere++) {
        if (!(sphere_id & visibility_mask)) continue;

        subVec3(&sphere->position, Ro, C);
        t = dotVec3(C, Rd);
        if (t > 0) {
            scaleVec3(Rd, t, I);
            isubVec3(I, C);
            r = sphere->radius;
            dt = r*r - squaredLengthVec3(I);

            if (dt > 0 && dt < closest_distance_squared) { // Inside the sphere
                d = sqrtf(dt);

                inner_hit_distance = t + d;
                outer_hit_distance = t - d;

                has_inner_hit = inner_hit_distance > 0 && inner_hit_distance < closest_hit.distance;
                has_outer_hit = outer_hit_distance > 0 && outer_hit_distance < closest_hit.distance;
                if (has_inner_hit ||
                    has_outer_hit) {
                    Sp = &sphere->position;
                    Sr = &sphere->rotation;

                    if (transparency_mask & sphere_id) {
                        if (has_outer_hit) {
                            d = outer_hit_distance + EPS;
                            current_hit.is_back_facing = false;
                            current_hit.uv = setRaySphereHit(Ro, Rd, P, N, Sp, Sr, d, false);
                            if (has_inner_hit && isTransparent(current_hit.uv)) {
                                d = inner_hit_distance - EPS;
                                current_hit.is_back_facing = true;
                                current_hit.uv = setRaySphereHit(Ro, Rd, P, N, Sp, Sr, d, true);
                                if (isTransparent(current_hit.uv)) continue;
                            }
                        } else {
                            d = inner_hit_distance - EPS;
                            current_hit.is_back_facing = true;
                            current_hit.uv = setRaySphereHit(Ro, Rd, P, N, Sp, Sr, d, true);
                            if (isTransparent(current_hit.uv)) continue;
                        }
                    } else {
                        current_hit.is_back_facing = !has_outer_hit;
                        d = has_outer_hit ? outer_hit_distance + EPS : inner_hit_distance - EPS;
                        current_hit.uv = setRaySphereHit(Ro, Rd, P, N, Sp, Sr, d, current_hit.is_back_facing);
                    }
                    closest_hit = current_hit;
                    closest_hit.material_id = sphere->material_id;
                    closest_hit.distance = d;
                    found = true;
                    if (check_any) break;
                }
            }
        }
    }

    if (found) ray->hit = closest_hit;

    return found;
}