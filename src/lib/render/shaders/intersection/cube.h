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
bool hitCubes(Cube *cubes, vec3 *Ro, vec3 *Rd, vec3 *P, vec3 *N, u8 *o_mat, f32 *o_dist) {
    vec3 hit_position, closest_hit_position, closest_hit_normal;
    f32 hit_distance, closest_hit_distance = *o_dist;
    u8 closest_hit_material = 0;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Cube *cube;
    Triangle* triangle;
    vec3 *p1, *p2, *p3, *n;
    for (u8 i = 0; i < CUBE_COUNT; i++) {
        cube = &cubes[i];
        for (u8 t = 0; t < 12; t++) {
            triangle = &cube->triangles[t];
            expandTriangle(triangle, cube->vertices, p1, p2, p3, n);

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
                    closest_hit_material = cube->material_id;

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