#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"
#include "plane.h"


#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitCubes(Cube *cubes, Indices *indices, Ray *ray, u8 visibility, bool check_any) {
    vec3 hit_position, hit_position_tangent;
    vec3 *Ro = ray->origin,
         *Rd = ray->direction;
    f32 x, y, distance, closest_distance = ray->hit.distance;
    u8 q, cude_id = 1;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Cube* cube = cubes;
    vec3 *v1, *n;

    for (u8 i = 0; i < CUBE_COUNT; i++, cube++, cude_id <<= 1) {
        if (!(cude_id & visibility)) continue;

        for (q = 0; q < 6; q++) {
            v1 = &cube->vertices[indices[q].v1];
            n = &cube->tangent_to_world[q].Z;
            if (hitPlane(v1, n, Rd, Ro, &distance)) {
                if (distance < closest_distance) {
                    scaleVec3(Rd, distance, &hit_position);
                    iaddVec3(&hit_position, Ro);

                    subVec3(&hit_position, v1, &hit_position_tangent);
                    imulVec3Mat3(&hit_position_tangent, &cube->world_to_tangent[q]);

                    x = hit_position_tangent.x;
                    y = hit_position_tangent.y;

                    if (x > 0 && y > 0 && x < 1 && y < 1) {
                        ray->hit.is_back_facing = false;
                        ray->hit.material_id = cube->node.geo.material_id;
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
//bool hitCubes(Cube *cubes, vec3 *Ro, vec3 *Rd, vec3 *P, vec3 *N, u8 *o_mat, f32 *o_dist) {
//    vec3 hit_position, closest_hit_position, closest_hit_normal;
//    f32 hit_distance, closest_hit_distance = *o_dist;
//    u8 closest_hit_material = 0;
//    bool found = false;
//
//    // Loop over all tetrahedra and intersect the ray against them:
//    Cube *cube;
//    Triangle* triangle;
//    vec3 *v1, *v2, *v3, *n;
//    for (u8 i = 0; i < CUBE_COUNT; i++) {
//        cube = &cubes[i];
//        for (u8 t = 0; t < 12; t++) {
//            triangle = &cube->triangles[t];
//            expandTriangle(triangle, cube->vertices, v1, v2, v3, n);
////            v1 = &cube->vertices[indices[t].v1];
////            v2 = &cube->vertices[indices[t].v2];
////            v3 = &cube->vertices[indices[t].v3];
////            n = &triangle->tangent_to_world.Z;
//            if (hitPlane(v1, n, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {
//                scaleVec3(Rd, hit_distance, &hit_position);
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
//                vec3 c1; crossVec3(&v1P, &v12, &c1);
//                vec3 c2; crossVec3(&v2P, &v23, &c2);
//                vec3 c3; crossVec3(&v3P, &v31, &c3);
//
//                if (dotVec3(n, &c1) > 0 &&
//                    dotVec3(n, &c2) > 0 &&
//                    dotVec3(n, &c3) > 0) {
//                    closest_hit_distance = hit_distance;
//                    closest_hit_position = hit_position;
//                    closest_hit_normal   = *n;
//                    closest_hit_material = cube->material_id;
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