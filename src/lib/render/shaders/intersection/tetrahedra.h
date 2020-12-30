#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/app.h"
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
bool hitTetrahedra(Tetrahedron *tetrahedra, TriangleIndices *indices, Ray *ray, u8 visibility_mask, bool check_any) {
    vec3 hit_position, hit_position_tangent;
    vec3 *Ro = ray->origin,
         *Rd = ray->direction;
    f32 x, y, distance, closest_distance = ray->hit.distance;
    u8 t, tetrahedron_id = 1;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron = tetrahedra;
    vec3 *v1, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tetrahedron++, tetrahedron_id <<= 1) {
        if (!(tetrahedron_id & visibility_mask)) continue;

        for (t = 0; t < 4; t++) {
            v1 = &tetrahedron->vertices[indices[t].v1];
            n = &tetrahedron->tangent_to_world[t].Z;
            if (hitPlane(v1, n, Rd, Ro, &distance)) {
                if (distance < closest_distance) {
                    scaleVec3(Rd, distance, &hit_position);
                    iaddVec3(&hit_position, Ro);

                    subVec3(&hit_position, v1, &hit_position_tangent);
                    imulVec3Mat3(&hit_position_tangent, &tetrahedron->world_to_tangent[t]);

                    x = hit_position_tangent.x;
                    y = hit_position_tangent.y;

                    if (x > 0 && y > 0 && y < (1 - x)) {
                        ray->hit.is_back_facing = false;
                        ray->hit.material_id = tetrahedron->node.geo.material_id;
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
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//bool hitTetrahedraExplicit(Tetrahedron *tetrahedra, TriangleIndices *indices, Ray *ray, bool check_any) {
//    vec3 hit_position;
//    vec3 *Ro = ray->origin,
//         *Rd = ray->direction;
//    f32 distance, closest_distance = ray->hit.distance;
//    u8 t, tetrahedron_id = 1, visibility_mask = ray->masks.visibility.tetrahedra;
//    bool found = false;
//
//    // Loop over all tetrahedra and intersect the ray against them:
//    Tetrahedron* tetrahedron;
//    Triangle *triangle;
//    vec3 *v1, *v2, *v3, *n;
//
//    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tetrahedron_id <<= 1) {
//        if (!(tetrahedron_id & visibility_mask)) continue;
//
//        tetrahedron = &tetrahedra[i];
//        for (t = 0; t < 4; t++) {
//            triangle = &tetrahedron->triangles[t];
//            v1 = &tetrahedron->vertices[indices[t].v1];
//            v2 = &tetrahedron->vertices[indices[t].v2];
//            v3 = &tetrahedron->vertices[indices[t].v3];
//            n = &triangle->tangent_to_world.Z;
//
//            if (hitPlane(v1, n, Rd, Ro, &distance) && distance < closest_distance) {
//                scaleVec3(Rd, distance, &hit_position);
//                iaddVec3(&hit_position, Ro);
//
//                vec3 v12; subVec3(v2, v1, &v12);
//                vec3 v23; subVec3(v3, v2, &v23);
//                vec3 v31; subVec3(v1, v3, &v31);
//
//                vec3 v1P; subVec3(&hit_position, v1, &v1P);
//                vec3 v2P; subVec3(&hit_position, v2, &v2P);
//                vec3 v3P; subVec3(&hit_position, v3, &v3P);
//
//                vec3 c1; crossVec3(&v12, &v1P, &c1);
//                vec3 c2; crossVec3(&v23, &v2P, &c2);
//                vec3 c3; crossVec3(&v31, &v3P, &c3);
//
//                if (dotVec3(n, &c1) > 0 &&
//                    dotVec3(n, &c2) > 0 &&
//                    dotVec3(n, &c3) > 0) {
//                    ray->hit.is_back_facing = false;
//                    ray->hit.material_id = tetrahedron->material_id;
//                    ray->hit.distance = closest_distance = distance;
//                    ray->hit.position = hit_position;
//                    ray->hit.normal = *n;
//                    found = true;
//                    if (check_any) break;
//                }
//            }
//        }
//    }
//
//    return found;
//}