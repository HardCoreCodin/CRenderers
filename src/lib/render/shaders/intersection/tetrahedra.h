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
bool hitTetrahedra(Tetrahedron *tetrahedra, IndexBuffers *_index_buffers, Ray *ray, bool check_any) {
    vec3 hit_position, hit_position_tangent;
    vec3 *Ro = ray->origin,
         *Rd = ray->direction;
    f32 x, y, distance, closest_distance = ray->hit.distance;
    u8 t, tetrahedron_id = 1, visibility_mask = ray->masks.visibility.tetrahedra;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron = tetrahedra;
    Triangle *triangle;
    vec3 *v1, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tetrahedron++, tetrahedron_id <<= 1) {
        if (!(tetrahedron_id & visibility_mask)) continue;

        triangle = tetrahedron->triangles;
        for (t = 0; t < 4; t++, triangle++) {
            v1 = &tetrahedron->vertices[_index_buffers->tetrahedron[t][0]];
            n = &triangle->tangent_to_world.Z;
            if (hitPlane(v1, n, Rd, Ro, &distance)) {
                if (distance < closest_distance) {
                    scaleVec3(Rd, distance, &hit_position);
                    iaddVec3(&hit_position, Ro);

                    subVec3(&hit_position, v1, &hit_position_tangent);
                    imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);
//                    imulVec3Mat3(&hit_position_tangent, &tetrahedron->scale);
//                    imulVec3Mat3(&hit_position_tangent, &tetrahedron->skew);

                    x = hit_position_tangent.x;
                    y = hit_position_tangent.y;

                    if (x > 0 && y > 0 && y < (1 - x)) {
                        ray->hit.is_back_facing = false;
                        ray->hit.material_id = tetrahedron->material_id;
                        ray->hit.distance = closest_distance = distance;
                        ray->hit.position = hit_position;
                        ray->hit.normal = *n;
                        found = true;
                        if (check_any) break;
                    }
                }
            }
        }
    }

    return found;
}

// ad - bc > 0
// a = Px
// b = Py
// c = v2x = 0
// d = v2y = 1
//
// Px*1 - Py*0 > 0
// Px > 0

// ad - bc > 0
// a = v3x = 1
// b = v3y = 0
// c = Px
// d = Py
//
// 1*Py - 0*Px > 0
// Py > 0

// ad - bc > 0
// a = v2x - v3x = 0 - 1 = -1
// b = v2y - v3y = 1 - 0 = 1
// c = Px - v3x = Px - 1
// d = Py - v3y = Py - 0
//
// -1*Py - 1*(Px - 1) > 0
// -Py > Px - 1
// Py < 1 - Px



#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitTetrahedra1(Tetrahedron *tetrahedra, Ray *ray, bool check_any) {
    vec3 hit_position;
    vec3 *Ro = ray->origin,
         *Rd = ray->direction;
    f32 distance, closest_distance = ray->hit.distance;
    u8 t, tetrahedron_id = 1, visibility_mask = ray->masks.visibility.tetrahedra;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron;
    Triangle *triangle;
    vec3 *v1, *v2, *v3, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tetrahedron_id <<= 1) {
        if (!(tetrahedron_id & visibility_mask)) continue;

        tetrahedron = &tetrahedra[i];
        for (t = 0; t < 4; t++) {
            triangle = &tetrahedron->triangles[t];
            expandTriangle(triangle, tetrahedron->vertices, v1, v2, v3, n);
            if (hitPlane(v1, n, Rd, Ro, &distance) && distance < closest_distance) {

                scaleVec3(Rd, distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                vec3 v12; subVec3(v2, v1, &v12);
                vec3 v23; subVec3(v3, v2, &v23);
                vec3 v31; subVec3(v1, v3, &v31);

                vec3 v1P; subVec3(&hit_position, v1, &v1P);
                vec3 v2P; subVec3(&hit_position, v2, &v2P);
                vec3 v3P; subVec3(&hit_position, v3, &v3P);

                vec3 c1; crossVec3(&v12, &v1P, &c1);
                vec3 c2; crossVec3(&v23, &v2P, &c2);
                vec3 c3; crossVec3(&v31, &v3P, &c3);

                if (dotVec3(n, &c1) > 0 &&
                    dotVec3(n, &c2) > 0 &&
                    dotVec3(n, &c3) > 0) {
                    ray->hit.is_back_facing = false;
                    ray->hit.material_id = tetrahedron->material_id;
                    ray->hit.distance = closest_distance = distance;
                    ray->hit.position = hit_position;
                    ray->hit.normal = *n;
//                    norm3(&ray->hit.normal);
//                    ray->hit.normal.x = -ray->hit.normal.x;
//                    ray->hit.normal.y = -ray->hit.normal.y;
//                    ray->hit.normal.z = -ray->hit.normal.z;
                    found = true;
                    if (check_any) break;
                }
            }
        }
    }

    return found;
}


//
//bool hitImplicitTetrahedra(Tetrahedron *tetrahedra, vec3 *Ro, vec3 *Rd, vec3 *P, vec3 *N, u8 *o_mat, f32 *o_dist) {
//    vec3 hit_position, closest_hit_position, hit_position_tangent, closest_hit_normal;
//    f32 x, y, hit_distance, closest_hit_distance = *o_dist;
//    u8 closest_hit_material = 0;
//    bool found = false;
//
//    // Loop over all tetrahedra and intersect the ray against them:
//    Tetrahedron* tetrahedron;
//    Triangle *triangle;
//    vec3 *p1, *n;
//
//    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
//        tetrahedron = &tetrahedra[i];
//        for (u8 t = 0; t < 4; t++) {
//            triangle = &tetrahedron->triangles[t];
//            expandTrianglePN(triangle, tetrahedron->vertices, p1, n);
//            if (hitPlane(p1, n, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {
//
//                scaleVec3(Rd, hit_distance, &hit_position);
//                iaddVec3(&hit_position, Ro);
//
//                subVec3(&hit_position, p1, &hit_position_tangent);
//                imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);
//                x = hit_position_tangent.x;
//                y = hit_position_tangent.y;
//
//                if (y > 0 && y < x*SQRT3 && y < (1 - x)*SQRT3) {
//                    closest_hit_distance = hit_distance;
//                    closest_hit_position = hit_position;
//                    closest_hit_normal   = *n;
//                    closest_hit_material = tetrahedron->material_id;
//
//                    found = true;
//                }
//            }
//        }
//    }
//
//    if (found) {
//        *N = closest_hit_normal;
//        *P = closest_hit_position;
//        *o_dist = closest_hit_distance;
//        *o_mat = closest_hit_material;
//    }
//
//    return found;
//}

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