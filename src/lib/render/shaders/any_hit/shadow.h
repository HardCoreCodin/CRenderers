#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/globals/scene.h"
#include "lib/globals/raytracing.h"
#include "lib/render/BVH.h"
#include "lib/math/math3D.h"
#include "../intersection/tetrahedra.h"
#include "../intersection/sphere.h"
#include "../intersection/plane.h"
#include "../intersection/AABB.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool inShadow(Scene *scene, vec3* Rd, vec3* Ro, f32 light_distance) {
    Ray ray;
    ray.origin = Ro;
    ray.direction = Rd;
    ray.hit.distance = light_distance;

    if (use_BVH) {
        vec3 Rd_rcp;
        Rd_rcp.x = 1.0f / Rd->x;
        Rd_rcp.y = 1.0f / Rd->y;
        Rd_rcp.z = 1.0f / Rd->z;
        ray.direction_rcp = &Rd_rcp;

        setRayMasksFromBVH(&ray, &ray_tracer.bvh);
        ray.masks.visibility.spheres &= ray_tracer.masks.shadowing.spheres;
        if (!ray.masks.visibility.spheres) return false;

//        bool hit;
//        hitAABB_init();
//        hitAABB_macro(hit, ray_tracer.bvh.nodes->aabb, ray);
//        if (!hit) return false;
//        if (!hitAABB(&ray_tracer.bvh.nodes->aabb, &ray)) return false;
    } else ray.masks.visibility = ray_tracer.masks.shadowing;
    ray.masks.transparency = ray_tracer.masks.transparency;

    return hitSpheres(scene->spheres, &ray, true) || hitTetrahedra(scene->tetrahedra, &ray, true);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool inShadowSimple(Tetrahedron *tetrahedra, vec3 *Rd, vec3* Ro, f32 light_distance) {
    vec3 hit_position;
    f32 hit_distance;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron;
    Triangle *triangle;
    vec3 *v1, *v2, *v3, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
        tetrahedron = &tetrahedra[i];
        for (u8 t = 0; t < 4; t++) {
            triangle = &tetrahedron->triangles[t];
            expandTriangle(triangle, tetrahedron->vertices, v1, v2, v3, n);
            if (hitPlane(v1, n, Rd, Ro, &hit_distance) && hit_distance < light_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                vec3 v12; subVec3(v2, v1, &v12);
                vec3 v23; subVec3(v3, v2, &v23);
                vec3 v31; subVec3(v1, v3, &v31);

                vec3 v1P; subVec3(&hit_position, v1, &v1P);
                vec3 v2P; subVec3(&hit_position, v2, &v2P);
                vec3 v3P; subVec3(&hit_position, v3, &v3P);

                vec3 c1; crossVec3(&v1P, &v12, &c1);
                vec3 c2; crossVec3(&v2P, &v23, &c2);
                vec3 c3; crossVec3(&v3P, &v31, &c3);

                if (dotVec3(n, &c1) > 0 &&
                    dotVec3(n, &c2) > 0 &&
                    dotVec3(n, &c3) > 0)
                    return true;
            }
        }
    }

//    Cube* last_cube = scene.cubes + scene.cube_count;
//    for (Cube* cube = scene.cubes; cube != last_cube; cube++) {
//        Triangle* last_triangle = cube->triangles + 12;
//        for (Triangle* triangle = cube->triangles; triangle != last_triangle; triangle++) {
//            if (hitPlane(triangle->p1, triangle->normal, Rd, Ro, &hit_distance) && hit_distance < light_distance) {
//
//                scaleVec3(Rd, hit_distance, &hit_position);
//                iaddVec3(&hit_position, Ro);
//
//                vec3 p1p2; subVec3(triangle->p2, triangle->p1, &p1p2);
//                vec3 p2p3; subVec3(triangle->p3, triangle->p2, &p2p3);
//                vec3 p3p1; subVec3(triangle->p1, triangle->p3, &p3p1);
//
//                vec3 p1P; subVec3(&hit_position, triangle->p1, &p1P);
//                vec3 p2P; subVec3(&hit_position, triangle->p2, &p2P);
//                vec3 p3P; subVec3(&hit_position, triangle->p3, &p3P);
//
//                vec3 c1; crossVec3(&p1P, &p1p2, &c1);
//                vec3 c2; crossVec3(&p2P, &p2p3, &c2);
//                vec3 c3; crossVec3(&p3P, &p3p1, &c3);
//
//                if (dotVec3(triangle->normal, &c1) > 0 &&
//                    dotVec3(triangle->normal, &c2) > 0 &&
//                    dotVec3(triangle->normal, &c3) > 0)
//                    return true;
//            }
//        }
//    }

//    f32 t, dt;
//    vec3 _I, *I = &_I,
//         _C, *C = &_C;
//
//    Sphere* last_sphere = scene.spheres + scene.sphere_count;
//    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
//        subVec3(&sphere->position, Ro, C);
//        t = dotVec3(C, Rd);
//        if (t > 0 && t < light_distance) {
//            scaleVec3(Rd, t, I);
//            isubVec3(I, C);
//            dt = sphere->radius*sphere->radius - squaredLengthVec3(I);
//            if (dt > 0 && t > sqrt(dt)) return true;
//        }
//    }

    return false;
}


#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool inShadowSimpleImplicit(Tetrahedron *tetrahedra, vec3 *Rd, vec3* Ro, f32 light_distance) {
    vec3 hit_position, hit_position_tangent;
    f32 x, y, hit_distance;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* tetrahedron;
    Triangle *triangle;
    vec3 *v1, *n;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
        tetrahedron = &tetrahedra[i];
        for (u8 t = 0; t < 4; t++) {
            triangle = &tetrahedron->triangles[t];
            expandTrianglePN(triangle, tetrahedron->vertices, v1, n);
            if (hitPlane(v1, n, Rd, Ro, &hit_distance) && hit_distance < light_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                subVec3(&hit_position, v1, &hit_position_tangent);
                imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);

                x = hit_position_tangent.x;
                y = hit_position_tangent.y;

                if (y > 0 && y < x*SQRT3 && y < (1 - x)*SQRT3) return true;
            }
        }
    }

    return false;
}