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
#include "../intersection/cube.h"
#include "../intersection/AABB.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool inShadow(Scene *scene, BVHNode *bvh_nodes, Masks *scene_masks, vec3* Rd, vec3* Ro, f32 light_distance) {
    vec3 Rd_rcp;
    Rd_rcp.x = 1.0f / Rd->x;
    Rd_rcp.y = 1.0f / Rd->y;
    Rd_rcp.z = 1.0f / Rd->z;

//    if (!hitAABB(&bvh_nodes->aabb.min, &bvh_nodes->aabb.max, Ro, &Rd_rcp))
//        return false;

    GeometryMasks visibility = getRayVisibilityMasksFromBVH(Ro, &Rd_rcp, bvh_nodes);
    visibility.cubes &= scene_masks->shadowing.cubes;
    visibility.spheres &= scene_masks->shadowing.spheres;
    visibility.tetrahedra &= scene_masks->shadowing.tetrahedra;
    if (!(visibility.cubes | visibility.spheres | visibility.tetrahedra))
        return false;

    Ray ray;
    ray.origin = Ro;
    ray.direction = Rd;
    ray.hit.distance = light_distance;

    if (visibility.spheres) {
        if (hitSpheres(scene->spheres, &ray, visibility.spheres, scene_masks->transparency.spheres, true))
            return true;
    }

    if (visibility.cubes) {
        if (hitCubes(scene->cubes, scene->cube_indices, &ray, visibility.cubes, true))
            return true;
    }

    if (visibility.tetrahedra) {
        if (hitTetrahedra(scene->tetrahedra, scene->tetrahedron_indices, &ray, visibility.tetrahedra, true))
            return true;
    }

    return false;
}

//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//bool inShadow(Scene *scene, BVHNode *bvh_nodes, Masks *scene_masks, vec3* Rd, vec3* Ro, f32 light_distance) {
//    vec3 RO = *Ro;
//    vec3 RD = *Rd;
//    vec3 v_min = bvh_nodes->aabb.min;
//    vec3 v_max = bvh_nodes->aabb.max;
//
//    f32 Ox = RO.x;
//    f32 Oy = RO.y;
//    f32 Oz = RO.x;
//
//    f32 Dx = 1.0f / RD.x;
//    f32 Dy = 1.0f / RD.y;
//    f32 Dz = 1.0f / RD.z;
//
//    f32 min_t_x = (v_min.x - Ox) * Dx;
//    f32 min_t_y = (v_min.y - Oy) * Dy;
//    f32 min_t_z = (v_min.z - Oz) * Dz;
//
//    f32 max_t_x = (v_max.x - Ox) * Dx;
//    f32 max_t_y = (v_max.y - Oy) * Dy;
//    f32 max_t_z = (v_max.z - Oz) * Dz;
//
//    f32 max_x = min_t_x > max_t_x ? min_t_x : max_t_x;
//    f32 max_y = min_t_y > max_t_y ? min_t_y : max_t_y;
//    f32 max_z = min_t_z > max_t_z ? min_t_z : max_t_z;
//
//    f32 min_x = min_t_x < max_t_x ? min_t_x : max_t_x;
//    f32 min_y = min_t_y < max_t_y ? min_t_y : max_t_y;
//    f32 min_z = min_t_z < max_t_z ? min_t_z : max_t_z;
//
//    f32 min_of_maxes = max_x < max_y ? max_x : max_y; min_of_maxes = max_z < min_of_maxes ? max_z : min_of_maxes;
//    f32 max_of_mins  = min_x > min_y ? min_x : min_y; max_of_mins  = min_z > max_of_mins  ? min_z : max_of_mins;
//    max_of_mins = 0.0f > max_of_mins ? 0.0f : max_of_mins;
//    if (min_of_maxes < max_of_mins) return false;
//
//    BVHNode *node = bvh_nodes;
//    u8 i, child, next_children, children = node->children;
//    u8 spheres = 0, cubes = 0, tetrahedra = 0;
//
//    bool cubes_found = false;
//    bool spheres_found = false;
//    bool tetrahedra_found = false;
//    bool hit_found = false;
//
//    while (children) { // Breadth-first_traversal
//        next_children = 0;
//        child = 1;
//
//        for (i = 0; i < 8; i++, child <<= (u8)1)
//            if (child & children) {
//                node = &bvh_nodes[i + 1];
//
//                v_min = node->aabb.min;
//                v_max = node->aabb.max;
//
//                min_t_x = (v_min.x - Ox) * Dx;
//                min_t_y = (v_min.y - Oy) * Dy;
//                min_t_z = (v_min.z - Oz) * Dz;
//
//                max_t_x = (v_max.x - Ox) * Dx;
//                max_t_y = (v_max.y - Oy) * Dy;
//                max_t_z = (v_max.z - Oz) * Dz;
//
//                max_x = min_t_x > max_t_x ? min_t_x : max_t_x;
//                max_y = min_t_y > max_t_y ? min_t_y : max_t_y;
//                max_z = min_t_z > max_t_z ? min_t_z : max_t_z;
//
//                min_x = min_t_x < max_t_x ? min_t_x : max_t_x;
//                min_y = min_t_y < max_t_y ? min_t_y : max_t_y;
//                min_z = min_t_z < max_t_z ? min_t_z : max_t_z;
//
//                min_of_maxes = max_x < max_y ? max_x : max_y; min_of_maxes = max_z < min_of_maxes ? max_z : min_of_maxes;
//                max_of_mins  = min_x > min_y ? min_x : min_y; max_of_mins  = min_z > max_of_mins  ? min_z : max_of_mins;
//                max_of_mins = 0.0f > max_of_mins ? 0.0f : max_of_mins;
//
//                if (min_of_maxes >= max_of_mins) {
//                    next_children |= node->children;
//                    switch (node->geo_type) {
//                        case GeoTypeCube:
//                            hit_found = true;
//                            cubes_found = true;
//                            cubes |= node->geo_ids;
//                            break;
//
//                        case GeoTypeSphere:
//                            hit_found = true;
//                            spheres_found = true;
//                            spheres |= node->geo_ids;
//                            break;
//
//                        case GeoTypeTetrahedron:
//                            hit_found = true;
//                            tetrahedra_found = true;
//                            tetrahedra |= node->geo_ids;
//                            break;
//                    }
//                }
//            }
//
//        children = next_children;
//    }
//    if (!hit_found) return false;
//
//    Ray ray;
//    ray.origin = Ro;
//    ray.direction = Rd;
//    ray.hit.distance = light_distance;
//
//    if (spheres_found) {
//        ray.masks.visibility.spheres = spheres;
//        if (hitSpheresSimple(scene->spheres, &ray, true))
//            return true;
//    }
//
//    if (cubes_found) {
//        if (hitCubes(scene->cubes, scene->cube_indices, &ray, cubes, true))
//            return true;
//    }
//
//    if (tetrahedra_found) {
//        if (hitTetrahedra(scene->tetrahedra, scene->tetrahedron_indices, &ray, tetrahedra, true))
//            return true;
//    }
//
//    return false;
//}

//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//bool inShadowSimple(Tetrahedron *tetrahedra, vec3 *Rd, vec3* Ro, f32 light_distance) {
//    vec3 hit_position;
//    f32 hit_distance;
//
//    // Loop over all tetrahedra and intersect the ray against them:
//    Tetrahedron* tetrahedron;
//    Triangle *triangle;
//    vec3 *v1, *v2, *v3, *n;
//
//    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
//        tetrahedron = &tetrahedra[i];
//        for (u8 t = 0; t < 4; t++) {
//            triangle = &tetrahedron->triangles[t];
//            expandTriangle(triangle, tetrahedron->vertices, v1, v2, v3, n);
//            if (hitPlane(v1, n, Rd, Ro, &hit_distance) && hit_distance < light_distance) {
//
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
//                    dotVec3(n, &c3) > 0)
//                    return true;
//            }
//        }
//    }

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
//
//    return false;
//}
//
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//bool inShadowSimpleImplicit(Tetrahedron *tetrahedra, vec3 *Rd, vec3* Ro, f32 light_distance) {
//    vec3 hit_position, hit_position_tangent;
//    f32 x, y, hit_distance;
//
//    // Loop over all tetrahedra and intersect the ray against them:
//    Tetrahedron* tetrahedron;
//    Triangle *triangle;
//    vec3 *v1, *n;
//
//    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
//        tetrahedron = &tetrahedra[i];
//        for (u8 t = 0; t < 4; t++) {
//            triangle = &tetrahedron->triangles[t];
//            expandTrianglePN(triangle, tetrahedron->vertices, v1, n);
//            if (hitPlane(v1, n, Rd, Ro, &hit_distance) && hit_distance < light_distance) {
//
//                scaleVec3(Rd, hit_distance, &hit_position);
//                iaddVec3(&hit_position, Ro);
//
//                subVec3(&hit_position, v1, &hit_position_tangent);
//                imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);
//
//                x = hit_position_tangent.x;
//                y = hit_position_tangent.y;
//
//                if (y > 0 && y < x*SQRT3 && y < (1 - x)*SQRT3) return true;
//            }
//        }
//    }
//
//    return false;
//}