#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"
#include "lib/globals/raytracing.h"
#include "plane.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitTetrahedraSimple(Tetrahedron *tetrahedra, Ray *ray) {
    vec3 hit_position, hit_position_tangent;
    vec3 *Ro = ray->origin,
         *Rd = ray->direction;
    f32 x, y, distance, closest_distance = ray->hit.distance;
    u8 t, tetrahedron_id = 1, visibility_mask = ray->masks.visibility[GEO_TYPE__TETRAHEDRON-1];
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron;
    Triangle *triangle;
    vec3 *p1, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tetrahedron_id <<= 1) {
        if (!(tetrahedron_id & visibility_mask)) continue;

        tetrahedron = &tetrahedra[i];
        for (t = 0; t < 4; t++) {
            triangle = &tetrahedron->triangles[t];
            expandTrianglePN(triangle, tetrahedron->vertices, p1, n);
            if (hitPlane(p1, n, Rd, Ro, &distance) && distance < closest_distance) {

                scaleVec3(Rd, distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                subVec3(&hit_position, p1, &hit_position_tangent);
                imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);
                x = hit_position_tangent.x;
                y = hit_position_tangent.y;

                if (y > 0 && y < x*SQRT3 && y < (1 - x)*SQRT3) {
                    ray->hit.is_back_facing = false;
                    ray->hit.material_id = tetrahedron->material_id;
                    ray->hit.distance = closest_distance = distance;
                    ray->hit.position = hit_position;
                    ray->hit.normal = *n;

                    found = true;
                }
            }
        }
    }

    return found;
}

bool hitImplicitTetrahedra(Tetrahedron *tetrahedra, vec3 *Ro, vec3 *Rd, vec3 *P, vec3 *N, u8 *o_mat, f32 *o_dist) {
    vec3 hit_position, closest_hit_position, hit_position_tangent, closest_hit_normal;
    f32 x, y, hit_distance, closest_hit_distance = *o_dist;
    u8 closest_hit_material = 0;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron;
    Triangle *triangle;
    vec3 *p1, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
        tetrahedron = &tetrahedra[i];
        for (u8 t = 0; t < 4; t++) {
            triangle = &tetrahedron->triangles[t];
            expandTrianglePN(triangle, tetrahedron->vertices, p1, n);
            if (hitPlane(p1, n, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                subVec3(&hit_position, p1, &hit_position_tangent);
                imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);
                x = hit_position_tangent.x;
                y = hit_position_tangent.y;

                if (y > 0 && y < x*SQRT3 && y < (1 - x)*SQRT3) {
                    closest_hit_distance = hit_distance;
                    closest_hit_position = hit_position;
                    closest_hit_normal   = *n;
                    closest_hit_material = tetrahedron->material_id;

                    found = true;
                }
            }
        }
    }

    if (found) {
        *N = closest_hit_normal;
        *P = closest_hit_position;
        *o_dist = closest_hit_distance;
        *o_mat = closest_hit_material;
    }

    return found;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitTetrahedra(Tetrahedron *tetrahedra, vec3 *Ro, vec3 *Rd, vec3 *P, vec3 *N, u8 *o_mat, f32 *o_dist) {
    vec3 hit_position, closest_hit_position, closest_hit_normal;
    f32 hit_distance, closest_hit_distance = *o_dist;
    u8 closest_hit_material = 0;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron;
    Triangle *triangle;
    vec3 *p1, *p2, *p3, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
        tetrahedron = &tetrahedra[i];
        for (u8 t = 0; t < 4; t++) {
            triangle = &tetrahedron->triangles[t];
            expandTriangle(triangle, tetrahedron->vertices, p1, p2, p3, n);
            if (hitPlane(p1, n, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                vec3 p1p2; subVec3(p2, p1, &p1p2);
                vec3 p2p3; subVec3(p3, p2, &p2p3);
                vec3 p3p1; subVec3(p1, p3, &p3p1);

                vec3 p1P; subVec3(&hit_position, p1, &p1P);
                vec3 p2P; subVec3(&hit_position, p2, &p2P);
                vec3 p3P; subVec3(&hit_position, p3, &p3P);

                vec3 c1; crossVec3(&p1P, &p1p2, &c1);
                vec3 c2; crossVec3(&p2P, &p2p3, &c2);
                vec3 c3; crossVec3(&p3P, &p3p1, &c3);

                if (dotVec3(n, &c1) > 0 &&
                    dotVec3(n, &c2) > 0 &&
                    dotVec3(n, &c3) > 0) {
                    closest_hit_distance = hit_distance;
                    closest_hit_position = hit_position;
                    closest_hit_normal   = *n;
                    closest_hit_material = tetrahedron->material_id;

                    found = true;
                }
            }
        }
    }

    if (found) {
        *N = closest_hit_normal;
        *P = closest_hit_position;
        *o_dist = closest_hit_distance;
        *o_mat = closest_hit_material;
    }

    return found;
}

// Implicit tetrahedra hit test:

// ad - bc > 0
// a = p3.x = 1/2
// b = p3.y = s3/2
// c = P.x
// d = P.y
//
// P.y > s3*P.x

// ad - bc > 0
// a = P.x
// b = P.y
// c = p2.x = 1
// d = P2.y = 0
//
// 0 < P.y

// ad - bc > 0
// a = P.x - p2.x = P.x - 1 = (P.x - 1)
// b = P.y - p2.y = P.y - 0 = P.y
// c = p3.x - p2.x = 1/2 - 1 = -1/2
// d = P3.y - p2.y = s3/2 - 0 = s3/2
//
// (P.x - 1)s3/2 > P.y*-1/2
// (P.x - 1)s3 > P.y*-1
// -1*(1 - P.x)s3 > -1*P.y
// (1 - P.x)s3 < P.y