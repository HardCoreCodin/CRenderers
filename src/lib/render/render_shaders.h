#pragma once

#include "lib/core/types.h"
#include "lib/globals/raytracing.h"
#include "lib/shapes/line.h"
#include "lib/shapes/bbox.h"
#include "lib/shapes/helix.h"
#include "lib/input/keyboard.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/controllers/camera_controller.h"
#include "lib/nodes/camera.h"
#include "lib/memory/allocators.h"

#include "lib/render/shaders/closest_hit/debug.h"
#include "lib/render/shaders/closest_hit/classic.h"
#include "lib/render/shaders/closest_hit/surface.h"
#include "lib/render/shaders/intersection/cube.h"
#include "lib/render/shaders/intersection/plane.h"
#include "lib/render/shaders/intersection/sphere.h"
#include "lib/render/shaders/intersection/tetrahedra.h"
#include "lib/render/shaders/ray_generation/primary_rays.h"
#include "BVH.h"
#include "SSB.h"

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
void renderBeauty(Ray *ray, Scene *scene, BVHNode *bvh_nodes, GeometryBounds *bounds, Masks *masks, u16 x, u16 y, Pixel* pixel) {
    tracePrimaryRay(ray, scene, bounds, masks, x, y);

    vec3 color;
    fillVec3(&color, 0);
//    shadeSurface(scene, bvh, masks, ray->hit.material_id, ray->direction,  &ray->hit.position, &ray->hit.normal, &color);
    shadeLambert(scene, bvh_nodes, masks, ray->direction, &ray->hit.position, &ray->hit.normal, &color);
//    shadePhong(scene, bvh, masks, ray->direction, &ray->hit.position, &ray->hit.normal &color);
//    shadeBlinn(scene, bvh, masks, ray->direction, &ray->hit.position, &ray->hit.normal, &color);
    setPixelColor(pixel, color);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void renderNormals(Ray *ray, Scene *scene, BVHNode *bvh_nodes, GeometryBounds *bounds, Masks *masks, u16 x, u16 y, Pixel* pixel) {
    tracePrimaryRay(ray, scene, bounds, masks, x, y);

    vec3 color;
    shadeDirection(&ray->hit.normal, &color);
    setPixelColor(pixel, color);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void renderDepth(Ray *ray, Scene *scene, BVHNode *bvh_nodes, GeometryBounds *bounds, Masks *masks, u16 x, u16 y, Pixel* pixel) {
    tracePrimaryRay(ray, scene, bounds, masks, x, y);

    vec3 color;
    shadeDepth(ray->hit.distance, &color);
    setPixelColor(pixel, color);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void renderUVs(Ray *ray, Scene *scene, BVHNode *bvh_nodes, GeometryBounds *bounds, Masks *masks, u16 x, u16 y, Pixel* pixel) {
    tracePrimaryRay(ray, scene, bounds, masks, x, y);

    vec3 color;
    shadeUV(ray->hit.uv, &color);
    setPixelColor(pixel, color);
}