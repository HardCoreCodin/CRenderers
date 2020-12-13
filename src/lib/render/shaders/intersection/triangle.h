#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"
#include "lib/globals/raytracing.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitTriangles(RayHit* closest_hit, Material** material_ptr) {
    vec3 *Rd = &closest_hit->ray_direction;
    vec3 *Ro = &closest_hit->ray_origin;
    vec3 hit_position, closest_hit_position, closest_hit_normal;
    f32 hit_distance, closest_hit_distance = closest_hit->distance;
    Material *closest_hit_material = NULL;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Triangle* last_triangle = scene.triangles + scene.triangle_count;
    for (Triangle* triangle = scene.triangles; triangle != last_triangle; triangle++) {
        if (hitPlane(triangle->p1, triangle->normal, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {

            scaleVec3(Rd, hit_distance, &hit_position);
            iaddVec3(&hit_position, Ro);

            vec3 p1p2; subVec3(triangle->p2, triangle->p1, &p1p2);
            vec3 p2p3; subVec3(triangle->p3, triangle->p2, &p2p3);
            vec3 p3p1; subVec3(triangle->p1, triangle->p3, &p3p1);

            vec3 p1P; subVec3(&hit_position, triangle->p1, &p1P);
            vec3 p2P; subVec3(&hit_position, triangle->p2, &p2P);
            vec3 p3P; subVec3(&hit_position, triangle->p3, &p3P);

            vec3 c1; crossVec3(&p1P, &p1p2, &c1);
            vec3 c2; crossVec3(&p2P, &p2p3, &c2);
            vec3 c3; crossVec3(&p3P, &p3p1, &c3);

            if (dotVec3(triangle->normal, &c1) > 0 &&
                dotVec3(triangle->normal, &c2) > 0 &&
                dotVec3(triangle->normal, &c3) > 0) {
                closest_hit_distance = hit_distance;
                closest_hit_position = hit_position;
                closest_hit_normal   = *triangle->normal;
//                closest_hit_material = tetrahedron->material;

                found = true;
            }
        }
    }

    if (found) {
        closest_hit->normal = closest_hit_normal;
        closest_hit->position = closest_hit_position;
        closest_hit->distance = closest_hit_distance;
        *material_ptr = closest_hit_material;
    }

    return found;
}