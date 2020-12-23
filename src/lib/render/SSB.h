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

    for (u8 i = 0; i < node_count; i++, node_id <<= 1, bounds++)
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
void setRayVisibilityMasksFromBounds(
        GeometeryMasks *ray_visibility_mask,
        GeometeryMasks *scene_visibility_mask,
        GeometryBounds *scene_geometry_bounds,
        u16 x,
        u16 y) {
    ray_visibility_mask->spheres    = getVisibilityMasksFromBounds(scene_geometry_bounds->spheres,    SPHERE_COUNT,      scene_visibility_mask->spheres,    x, y);
    ray_visibility_mask->tetrahedra = getVisibilityMasksFromBounds(scene_geometry_bounds->tetrahedra, TETRAHEDRON_COUNT, scene_visibility_mask->tetrahedra, x, y);
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
            bottom = max(bottom, -1); bottom += 1;
            top    = min(top,    +1); top    += 1;
            left   = max(left,   -1); left   += 1;
            right  = min(right,  +1); right  += 1;

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
    u8 visible_nodes = 0;
    u8 visibility_mask = 0;
    u8 transparency_mask = masks->transparency.spheres;
    u8 node_id = 1;

    Sphere *s = scene->spheres;
    vec3 *p = ssb->view_positions.spheres;
    Bounds2Di *b = ssb->bounds.spheres;

    f32 r, x, y, z;
    for (u8 i = 0; i < SPHERE_COUNT; i++, node_id <<= 1, s++, p++, b++) {
        r = s->radius;
        x = p->x;
        y = p->y;
        z = p->z;

        if ((transparency_mask & node_id ? (z > -r) : (z > r)) &&
            computeSSB(b, x, y, z, r, focal_length)) {

            visibility_mask |= node_id;
            visible_nodes++;
        }
    }
    masks->visibility.spheres =  visibility_mask;
    ray_tracer.stats.visible_nodes[GEO_TYPE__SPHERE-1] = visible_nodes;

    visible_nodes = 0;
    visibility_mask = 0;
    node_id = 1;

    Tetrahedron *t = scene->tetrahedra;
    p = ssb->view_positions.tetrahedra;
    b = ssb->bounds.tetrahedra;

    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, t++, b++, p++, node_id <<= 1) {
        r = t->radius;
        x = p->x;
        y = p->y;
        z = p->z;

        if (computeSSB(b, x, y, z, r, focal_length)) {
            visibility_mask |= node_id;
            visible_nodes++;
        }
    }

    masks->visibility.tetrahedra = visibility_mask;
    ray_tracer.stats.visible_nodes[GEO_TYPE__TETRAHEDRON-1] = visible_nodes;

#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_sphere_view_bounds, scene->sphere_view_bounds, sizeof(Bounds2Di) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_masks, scene->masks, sizeof(Masks), 0, cudaMemcpyHostToDevice));
#endif
}

//void updateSceneMasksOld(Scene* scene) {
//    ray_tracer.active_sphere_count = 0;
//    f32 x, y, z, r, w, h, f, ff,
//            left, right, top, bottom,
//            one_over_focal_length = 1.0f / current_camera_controller->camera->focal_length;
//    Sphere* sphere;
//    vec3 *position;
//    Bounds2Di *bounds;
//
//    bool has_transparency;
//    scene->masks->visibility[GEO_TYPE__SPHERE-1] = 0;
//    scene->masks->has_transparency[GEO_TYPE__SPHERE-1] = 0;
//    u8 sphere_id = 1;
//    for (u8 i = 0; i < SPHERE_COUNT; i++, sphere_id <<= (u8)1) {
//        sphere = &scene->spheres[i];
//        position = &scene->sphere_view_positions[i];
//        r = sphere->radius;
//        x = position->x;
//        y = position->y;
//        z = position->z;
//
//        has_transparency = scene->materials[sphere->material_id].uses & (u8) TRANSPARENCY;
//        if (has_transparency) scene->masks->has_transparency[GEO_TYPE__SPHERE-1] |= sphere_id;
//        if (has_transparency ? (z > -r) : (z > r)) {
//            w = z * one_over_focal_length;
//            left = x - r;
//            right = x + r;
//            if ((x > 0 && left < +w) ||
//                (x < 0 && right > -w)) {
//                h = w * frame_buffer.height_over_width;
//                top = y + r;
//                bottom = y - r;
//                if ((y > 0 && bottom < +h) ||
//                    (y < 0 && top > -h)) {
//                    f = (f32)frame_buffer.width / (w + w);
//
//                    ff = f / (z * r/2 * current_camera_controller->camera->focal_length);
//                    left -= ff;
//                    right += ff;
//                    bottom -= ff;
//                    top += ff;
//                    scene->masks->visibility[GEO_TYPE__SPHERE-1] |= sphere_id;
//                    bounds = &scene->sphere_view_bounds[i];
//                    bounds->x_range.min = (u16)(f * (w + max(-w, left)));
//                    bounds->x_range.max = (u16)(f * (w + min(+w, right)));
//                    bounds->y_range.max = frame_buffer.height - (u16)(f * (h + max(-h, bottom)));
//                    bounds->y_range.min = frame_buffer.height - (u16)(f * (h + min(+h, top)));
//                    ray_tracer.active_sphere_count++;
//                }
//            }
//        }
//    }
//
//#ifdef __CUDACC__
//    gpuErrchk(cudaMemcpyToSymbol(d_sphere_view_bounds, scene->sphere_view_bounds, sizeof(Bounds2Di) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
//    gpuErrchk(cudaMemcpyToSymbol(d_masks, scene->masks, sizeof(Masks), 0, cudaMemcpyHostToDevice));
//#endif
//}

void drawSSB(SSB* ssb, Pixel *pixel) {
    Bounds2Di *bounds = ssb->bounds.spheres;
    for (u8 i = 0; i < SPHERE_COUNT; i++, bounds++) {
        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.min, pixel);
        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.max, pixel);
        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.min, pixel);
        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.max, pixel);
    }
    bounds = ssb->bounds.tetrahedra;
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, bounds++) {
        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.min, pixel);
        drawHLine2D(bounds->x_range.min, bounds->x_range.max, bounds->y_range.max, pixel);
        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.min, pixel);
        drawVLine2D(bounds->y_range.min, bounds->y_range.max, bounds->x_range.max, pixel);
    }
}


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

 v = zz - rr
 t/z = 1/v * (yz -/+ r*sqrt(yy + v)


 r / h = z / H
 t + h = y
 HH = zz + tt

 t = y - h

 rH = zh
 hz = rH
 h = Hr/z

 t = y - h
 t = y - Hr/z
 Hr/z = y - t
 H = z/r(y - t)
 HH = zz/rr(y - t)(y - t)

 HH = zz + tt

 zz + tt = zz/rr(y - t)(y - t)
 zz + tt = zz/rr(yy - 2yt + tt)
 zz + tt = yyzz/rr - 2ytzz/rr + ttzz/rr
 zz + tt - yyzz/rr + 2ytzz/rr - ttzz/rr = 0
 tt - ttzz/rr + 2ytzz/rr + zz - yyzz/rr = 0
 (1 - zz/rr)tt + (2yzz/rr)t + zz(1 - yy/rr) = 0

 a = 1 - zz/rr
 b = 2yzz/rr
 c = zz(1 - yy/rr)

 t = (-b +/- sqrt(bb - 4ac)) / 2a

 bb - 4ac = 2yzz/rr * 2yzz/rr -4( (1 - zz/rr) * (zz(1 - yy/rr)) )
 bb - 4ac = 2 * yzz/rr * 2 * yzz/rr -4( (1 - zz/rr) * zz(1 - yy/rr) )
 bb - 4ac = 4*(yzz/rr * yzz/rr) -4*( (1 - zz/rr) * zz(1 - yy/rr) )
 bb - 4ac = 4*( yzz/rr * yzz/rr -    (1 - zz/rr) * zz(1 - yy/rr) )

 f = yzz/rr * yzz/rr - (1 - zz/rr) * zz(1 - yy/rr)

 bb - 4ac = 4 * f

 t = (-b +/- sqrt(4 * f)) / 2a
 t = (-b +/- sqrt(4) * sqrt(f)) / 2a
 t = (-b +/- 2 * sqrt(f)) / 2a
 t = (2*-yzz/rr +/- 2*sqrt(f)) / 2a
 t = 2*(-yzz/rr +/- sqrt(f)) / 2*a
 t = (-yzz/rr +/- sqrt(f)) / a

 f = yzz/rr * yzz/rr - (1 - zz/rr) * zz(1 - yy/rr)
 f = zz(y/rr) * zz(y/rr) - zz(1 - yy/rr)(1 - zz/rr)
 f = zz * ( y/rr * y/rr - (1 - yy/rr)(1 - zz/rr))

 f` = ( y/rr * y/rr - (1 - yy/rr)(1 - zz/rr))
 f = zz * f`

 t = (-yzz/rr +/- sqrt(f)) / a
 t = (-yzz/rr +/- sqrt(zz * f`)) / a
 t = (-yzz/rr +/- sqrt(zz) * sqrt(f`)) / a
 t = (z * -yz/rr +/- z * sqrt(f`)) / a
 t = z(-yz/rr +/- sqrt(f`)) / a

 f` = y/rr * y/rr -1*(1 - yy/rr)(1 - zz/rr)
 f` = y/rr * y/rr + (-1*(1 - yy/rr))(1 - zz/rr)
 f` = y/rr * y/rr + (+1*(-1 + yy/rr))(1 - zz/rr)
 f` = y/rr * y/rr + (yy/rr - 1)(1 - zz/rr)
 f` = y*(1/rr) * y/rr + yy*(1/rr) - yy*(1/rr)*zz/rr + zz*(1/rr) - (1/rr)*rr
 f` = (1/rr)*(yy/rr + yy - yy*zz/rr + zz - rr)

 f`` = yy/rr + yy - yy*zz/rr + zz - rr
 f` = (1/rr)*f``
 t = z(-yz/rr +/- sqrt(f`)) / a
 t = z(-yz/rr +/- sqrt((1/rr)*f``)) / a
 t = z(-yz/rr +/- sqrt(1/rr)*sqrt(f``)) / a
 t = z((1/r)-yz/r +/- 1/r * sqrt(f``)) / a
 t = (1/r)z(-yz/r +/- sqrt(f``)) / a
 t = z(-yz/r +/- sqrt(f``)) / ra

 f`` = yy/rr + yy - yy*zz/rr + zz - rr
 f`` = yy/rr + yy - yy*zz/rr + yy*zz/yy - yy*rr/yy
 f`` = yy * (1/rr + 1 - zz/rr + zz/yy - rr/yy)

 f``` = 1/rr + 1 - zz/rr + zz/yy - rr/yy
 f`` = yy * f```

 t = z(-yz/r +/- sqrt(f``)) / ra
 t = z(-yz/r +/- sqrt(yy * f```) / ra
 t = z(-yz/r +/- sqrt(yy) * sqrt(f```) / ra
 t = z(-yz/r +/- y * sqrt(f```) / ra
 t = z(y*-z/r +/- y*sqrt(f```) / ra
 t = zy(-z/r +/- sqrt(f```) / ra
 t = (-z/r +/- sqrt(f```)  * zy/ra

 f``` = 1 + 1/rr - (1/rr)*zz + zz(1/yy) - rr(1/yy(
 f``` = 1 + (1 - zz)/rr + (zz - rr)/yy

 1 - zz/rr

 zy/ra = zy/r(1 - zz/rr)
 zy/ra = zy/(r - zz/r)

 */