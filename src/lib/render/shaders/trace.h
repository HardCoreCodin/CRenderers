#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"
#include "lib/globals/raytracing.h"
#include "lib/render/BVH.h"

#include "intersection/tetrahedra.h"
#include "intersection/sphere.h"
#include "intersection/plane.h"
#include "intersection/cube.h"
#include "intersection/AABB.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void tracePrimaryRay(Ray *ray, Scene *scene, GeometryBounds *bounds, Masks *scene_masks, u16 x, u16 y) {
    ray->hit.uv.x = ray->hit.uv.y = 1;
    ray->hit.distance = MAX_DISTANCE;

    u8 visibility;

    hitPlanes(scene->planes, ray);

    visibility = getVisibilityMasksFromBounds(bounds->spheres, SPHERE_COUNT, scene_masks->visibility.spheres, x, y);
    if (visibility) hitSpheres(scene->spheres, ray, visibility, scene_masks->transparency.spheres, false);

    visibility = getVisibilityMasksFromBounds(bounds->cubes, CUBE_COUNT, scene_masks->visibility.cubes, x, y);
    if (visibility) hitCubes(scene->cubes, scene->cube_indices, ray, visibility, false);

    visibility = getVisibilityMasksFromBounds(bounds->tetrahedra, TETRAHEDRON_COUNT, scene_masks->visibility.tetrahedra, x, y);
    if (visibility) hitTetrahedra(scene->tetrahedra, scene->tetrahedron_indices, ray, visibility, false);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void traceSecondaryRay(Ray *ray, Scene *scene, BVHNode *bvh_nodes, Masks *scene_masks) {
    ray->hit.uv.x = ray->hit.uv.y = 1;
    ray->hit.distance = MAX_DISTANCE;

    vec3 Rd_rcp;
    Rd_rcp.x = 1.0f / ray->direction->x;
    Rd_rcp.y = 1.0f / ray->direction->y;
    Rd_rcp.z = 1.0f / ray->direction->z;

//    if (!hitAABB(&bvh_nodes->aabb.min, &bvh_nodes->aabb.max, Ro, &Rd_rcp))
//        return false;


    hitPlanes(scene->planes, ray);
    GeometryMasks visibility = getRayVisibilityMasksFromBVH(ray->origin, &Rd_rcp, bvh_nodes);

    visibility.cubes &= scene_masks->visibility.cubes;
    visibility.spheres &= scene_masks->visibility.spheres;
    visibility.tetrahedra &= scene_masks->visibility.tetrahedra;

    if (visibility.spheres) hitSpheres(scene->spheres, ray, visibility.spheres, scene_masks->transparency.spheres, true);
    if (visibility.cubes) hitCubes(scene->cubes, scene->cube_indices, ray, visibility.cubes, true);
    if (visibility.tetrahedra) hitTetrahedra(scene->tetrahedra, scene->tetrahedron_indices, ray, visibility.tetrahedra, true);
}


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