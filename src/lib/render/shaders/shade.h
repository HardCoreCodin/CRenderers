#pragma once

#include "lib/core/types.h"
#include "lib/core/color.h"
#include "lib/globals/raytracing.h"

#include "lib/render/shaders/closest_hit/debug.h"
#include "lib/render/shaders/closest_hit/classic.h"
#include "lib/render/shaders/closest_hit/surface.h"
#include "lib/render/shaders/closest_hit/reflection.h"

#include "lib/render/BVH.h"
#include "lib/render/SSB.h"

#include "trace.h"

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
//    shadeReflection(scene, bvh_nodes, masks, ray->hit.material_id, ray->direction,  &ray->hit.position, &ray->hit.normal, 0, &color);
    shadeSurface(scene, bvh_nodes, masks, ray->hit.material_id, ray->direction,  &ray->hit.position, &ray->hit.normal, &color);
//    shadeLambert(scene, bvh_nodes, masks, ray->direction, &ray->hit.position, &ray->hit.normal, &color);
//    shadePhong(scene, bvh_nodes, masks, ray->direction, &ray->hit.position, &ray->hit.normal, &color);
//    shadeBlinn(scene, bvh_nodes, masks, ray->direction, &ray->hit.position, &ray->hit.normal, &color);
    setPixelBakedToneMappedColor(pixel, color);
//    setPixelGammaCorrectedColor(pixel, color);
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