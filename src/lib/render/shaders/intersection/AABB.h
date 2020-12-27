#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/raytracing.h"


#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitAABB(AABB *aabb, Ray *ray) {
    vec3 *aabb_min = &aabb->min, *Ro = ray->origin,
         *aabb_max = &aabb->max, *RD_rcp = ray->direction_rcp;
    f32 min_x = aabb_min->x, max_x = aabb_max->x, Ox = Ro->x, Dx = RD_rcp->x, min_t_x = (min_x - Ox) * Dx, max_t_x = (max_x - Ox) * Dx,
        min_y = aabb_min->y, max_y = aabb_max->y, Oy = Ro->y, Dy = RD_rcp->y, min_t_y = (min_y - Oy) * Dy, max_t_y = (max_y - Oy) * Dy,
        min_z = aabb_min->z, max_z = aabb_max->z, Oz = Ro->z, Dz = RD_rcp->z, min_t_z = (min_z - Oz) * Dz, max_t_z = (max_z - Oz) * Dz;

    return min(min(max(min_t_x, max_t_x), max(min_t_y, max_t_y)), max(min_t_z, max_t_z)
    ) >= max(0.0f, max(max(min(min_t_x, max_t_x), min(min_t_y, max_t_y)), min(min_t_z, max_t_z)));
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool hitAABBold(AABB *aabb, Ray *ray) {
    f32 tmin, tmax;
    vec3 min_t, max_t;
    subVec3(&aabb->min, ray->origin, &min_t); imulVec3(&min_t, ray->direction_rcp);
    subVec3(&aabb->max, ray->origin, &max_t); imulVec3(&max_t, ray->direction_rcp);
    tmin = max(max(min(min_t.x, max_t.x), min(min_t.y, max_t.y)), min(min_t.z, max_t.z));
    tmax = min(min(max(min_t.x, max_t.x), max(min_t.y, max_t.y)), max(min_t.z, max_t.z));

    return tmax >= max(0.0f, tmin);
}

#define hitAABB_init() f32 min_t_x, min_t_y, min_t_z, max_t_x, max_t_y, max_t_z
#define hitAABB_components(hit, min_x, min_y, min_z, max_x, max_y, max_z, Ox, Oy, Oz, Dx, Dy, Dz) \
    min_t_x = (min_x - Ox) * Dx; \
    min_t_y = (min_y - Oy) * Dy; \
    min_t_z = (min_z - Oz) * Dz; \
    max_t_x = (max_x - Ox) * Dx; \
    max_t_y = (max_y - Oy) * Dy; \
    max_t_z = (max_z - Oz) * Dz; \
    hit = min(min(max(min_t_x, max_t_x), max(min_t_y, max_t_y)), max(min_t_z, max_t_z)) >= \
max(0.0f, max(max(min(min_t_x, max_t_x), min(min_t_y, max_t_y)), min(min_t_z, max_t_z)))

#define hitAABB_vec3pointers(hit, v_min, v_max, O, D) \
    hit = hitAABB_components(hit, v_min.x, v_min.y, v_min.z, v_max.x, v_max.y, v_max.z, O->x, O->y, O->z, D->x, D->y, D->z)

#define hitAABB_macro(hit, aabb, ray) hitAABB_vec3pointers(hit, aabb.min, aabb.max, ray.origin, ray.direction_rcp)

void setAABB(AABB *aabb, f32 r, vec3 *p) {
    f32 x = p->x, y = p->y, z = p->z;

    aabb->min.x = x - r;
    aabb->min.y = y - r;
    aabb->min.z = z - r;

    aabb->max.x = x + r;
    aabb->max.y = y + r;
    aabb->max.z = z + r;
}