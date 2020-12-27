#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/globals/scene.h"
#include "lib/globals/camera.h"
#include "lib/globals/display.h"
#include "lib/globals/raytracing.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
u8 getVisibilityMasksFromBounds(Bounds2Di *bounds, u8 node_count, u8 node_visibility, u16 x, u16 y) {
    u8 ray_visibility_mask = 0;
    u8 node_id = 1;

    for (u8 i = 0; i < node_count; i++, node_id <<= (u8)1, bounds++)
        if (node_visibility & node_id &&
            x >= bounds->x_range.min &&
            x <= bounds->x_range.max &&
            y >= bounds->y_range.min &&
            y <= bounds->y_range.max)
            ray_visibility_mask |= node_id;

    return ray_visibility_mask;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setRayVisibilityMasksFromBounds(Masks *ray_masks, Masks *scene_masks, GeometryBounds *bounds, u16 x, u16 y) {
    ray_masks->visibility.spheres    = getVisibilityMasksFromBounds(bounds->spheres,    SPHERE_COUNT,      scene_masks->visibility.spheres,    x, y);
    ray_masks->visibility.tetrahedra = getVisibilityMasksFromBounds(bounds->tetrahedra, TETRAHEDRON_COUNT, scene_masks->visibility.tetrahedra, x, y);
}

bool computeSSB(Bounds2Di *bounds, f32 x, f32 y, f32 z, f32 r, f32 focal_length) {
/*
 h = y - t
 HH = zz + tt

 r/z = h/H
 rH = zh
 rrHH = zzhh
 rr(zz + tt) = zz(y -t)(y - t)
 rrzz + rrtt = zz(yy -2ty + tt)
 rrzz + rrtt = zzyy -2tyzz + ttzz
 rrtt - zztt + (2yzz)t + rrzz - zzyy = 0
 (rr - zz)tt + (2yzz)t + zz(rr - yy) = 0

 a = rr - zz
 b = 2yzz
 c = zz(rr - yy)

 t = -b/2a +/- sqrt(bb - 4ac)/2a
 t = -2yzz/2(rr - zz) +/- sqrt(2yzz2yzz - 4(rr - zz)zz(rr - yy))/2(rr - zz)
 t = -yzz/(rr - zz) +/- sqrt(4yzzyzz - 4(rr - zz)zz(rr - yy))/2(rr - zz)
 t = -yzz/(rr - zz) +/- sqrt(yyzzzz - zz(rr - zz)(rr - yy))/(rr - zz)
 t = -yzz/(rr - zz) +/- sqrt(zz(yyzz - (rr - zz)(rr - yy)))/(rr - zz)
 t = -yzz/(rr - zz) +/- z*sqrt(yyzz - (rr - zz)(rr - yy))/(rr - zz)
 t = -yzz/(rr - zz) +/- z*sqrt(yyzz - (rr - zz)(rr - yy))/(rr - zz)

 t/z = 1/(rr - zz) * (-yz +/- sqrt(yyzz - (rr - zz)(rr - yy)))
 t/z = 1/(rr - zz) * (-yz +/- sqrt(yyzz - rr*rr + zz*rr + rr*yy - zz*yy))
 t/z = 1/(rr - zz) * (-yz +/- sqrt(yyzz - zz*yy - rr*rr + zz*rr + rr*yy))
 t/z = 1/(rr - zz) * (-yz +/- sqrt(0 - rr*rr + zz*rr + rr*yy))
 t/z = 1/(rr - zz) * (-yz +/- sqrt(rr(yy - rr + zz))
 t/z = -1/(rr - zz) * -(-yz +/- r*sqrt(zz + yy - rr)
 t/z = 1/(zz - rr) * (yz -/+ r*sqrt(yy + zz - rr)

  t/z = 1/(zz - rr) * (yz -/+ r*sqrt(yy + zz - rr)
  den = zz - rr
  sqr = r * sqrt(yy + den)

  t/z = 1/den * (yz -/+ sqr)
  s/fl = t/z
  s = fl*t/z
  s = fl/den * (yz -/+ sqr)

  f = fl/den
  s = f(yx -/+ sqr)

  s1 = f(yz - sqr)
  s2 = f(yz + sqr)
*/
    bounds->x_range.min = frame_buffer.width + 1;
    bounds->x_range.max = frame_buffer.width + 1;
    bounds->y_range.max = frame_buffer.height + 1;
    bounds->y_range.min = frame_buffer.height + 1;

    f32 den = z*z - r*r;
    f32 factor = focal_length / den;

    f32 xz = x * z;
    f32 sqr = r * sqrtf(x*x + den);
    f32 left  = factor*(xz - sqr);
    f32 right = factor*(xz + sqr);
    if (left < 1 && right > -1) {
        factor *= frame_buffer.width_over_height;

        f32 yz = y * z;
        sqr = r * sqrtf(y*y + den);
        f32 bottom = factor*(yz - sqr);
        f32 top    = factor*(yz + sqr);
        if (bottom < 1 && top > -1) {
            bottom = bottom > -1 ? bottom : -1; bottom += 1;
            top    = top < 1 ? top : 1; top    += 1;
            left   = left > -1 ? left : -1; left   += 1;
            right  = right < 1 ? right : 1; right  += 1;

            top    = 2 - top;
            bottom = 2 - bottom;

            bounds->x_range.min = (u16)(frame_buffer.h_width * left);
            bounds->x_range.max = (u16)(frame_buffer.h_width * right);
            bounds->y_range.max = (u16)(frame_buffer.h_height * bottom);
            bounds->y_range.min = (u16)(frame_buffer.h_height * top);
            return true;
        }
    }

    return false;
}

void updateSceneMasks(Scene* scene, SSB* ssb, Masks *masks, f32 focal_length) {
    u8 node_id, geo_count, transparency_mask, *visibility_mask;
    f32 r, z;
    vec3 *p;
    Bounds2Di *b;
    Node *node, **node_ptr;

    for (u8 geo_type = 0; geo_type < GEO_TYPE_COUNT; geo_type++) {
        switch (geo_type) {
            case GeoTypeCube:
                geo_count = CUBE_COUNT;
                node_ptr = scene->node_ptrs.cubes;
                transparency_mask = masks->transparency.cubes;
                visibility_mask = &masks->visibility.cubes;
                p = ssb->view_positions.cubes;
                b = ssb->bounds.cubes;
                break;
            case GeoTypeSphere:
                geo_count = SPHERE_COUNT;
                node_ptr = scene->node_ptrs.spheres;
                transparency_mask = masks->transparency.spheres;
                visibility_mask = &masks->visibility.spheres;
                p = ssb->view_positions.spheres;
                b = ssb->bounds.spheres;
                break;
            case GeoTypeTetrahedron:
                geo_count = TETRAHEDRON_COUNT;
                node_ptr = scene->node_ptrs.tetrahedra;
                transparency_mask = masks->transparency.tetrahedra;
                visibility_mask = &masks->visibility.tetrahedra;
                p = ssb->view_positions.tetrahedra;
                b = ssb->bounds.tetrahedra;
                break;
            default:
                continue;
        }

        *visibility_mask = 0;
        node_id = 1;

        for (u8 i = 0; i < geo_count; i++, p++, b++, node_ptr++, node_id <<= (u8)1) {
            node = *node_ptr;
            r = node->radius;
            z = p->z;

            if ((transparency_mask & node_id ? (z > -r) : (z > r)) &&
                computeSSB(b, p->x, p->y, p->z, r, focal_length)) {

                *visibility_mask |= node_id;
            }
        }
    }
    masks->visibility.cubes = masks->visibility.spheres = 0;

#ifdef __CUDACC__
    copyMasksFromCPUtoGPU(masks);
    copySSBBoundsFromCPUtoGPU(bounds);
#endif
}

void drawSSB(SSB* ssb, Pixel *pixel) {
    Bounds2Di *bounds = ssb->bounds.spheres;
//    for (u8 i = 0; i < SPHERE_COUNT; i++, bounds++) {
//        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.min, pixel);
//        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.max, pixel);
//        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.min, pixel);
//        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.max, pixel);
//    }

    bounds = ssb->bounds.tetrahedra;
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, bounds++) {
        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.min, pixel);
        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.max, pixel);
        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.min, pixel);
        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.max, pixel);
    }

//    bounds = ssb->bounds[GeoTypeCube];
//    for (u8 i = 0; i < CUBE_COUNT; i++, bounds++) {
//        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.min, pixel);
//        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.max, pixel);
//        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.min, pixel);
//        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.max, pixel);
//    }
}