#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/globals/scene.h"
#include "lib/math/math3D.h"
#include "../trace.h"
#include "common.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeLambert(Scene *scene, BVHNode *bvh_nodes, Masks *masks, vec3 *Rd, vec3 *P, vec3 *N, vec3 *out_color) {
    f32 d, d2;
    vec3 L;
    vec3 light_color,
         color = scene->ambient_light->color;

    if (dotVec3(N, Rd) > 0) iscaleVec3(N, -1);

    PointLight *light;
    for (u8 i = 0; i < POINT_LIGHT_COUNT; i++) {
        light = &scene->point_lights[i];
        subVec3(&light->position, P, &L);

        d2 = squaredLengthVec3(&L);
        d = sqrtf(d2);
        iscaleVec3(&L, 1.0f / d);

        if (inShadow(scene, bvh_nodes, masks, &L, P, d)) continue;

        scaleVec3(&light->color,light->intensity * sdot(N, &L) / d2, &light_color);
        iaddVec3(&color, &light_color);
    }

    iaddVec3(out_color, &color);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadePhong(Scene *scene, BVHNode *bvh_nodes, Masks *masks, vec3 *Rd, vec3 *P, vec3 *N, vec3 *out_color) {
    vec3 light_color, color = scene->ambient_light->color;
    vec3 _rl, *RLd = &_rl;
    vec3 _l, *L = &_l;
    f32 d, d2, li, diff, spec;
    reflect(Rd, N, -sdotInv(N, Rd), RLd);

    PointLight *light;
    for (u8 i = 0; i < POINT_LIGHT_COUNT; i++) {
        light = &scene->point_lights[i];
        subVec3(&light->position, P, L);

        d2 = squaredLengthVec3(L);
        d = sqrtf(d2);
        iscaleVec3(L, 1.0f / d);
        if (inShadow(scene, bvh_nodes, masks, L, P, d)) continue;

        li = light->intensity / d2;
        diff = li * sdot(N, L);
        spec = li * powf(sdot(RLd, L), 4);

        scaleVec3(&light->color, diff + spec, &light_color);
        iaddVec3(&color, &light_color);
    }

    iaddVec3(out_color, &color);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeBlinn(Scene *scene, BVHNode *bvh_nodes, Masks *masks, vec3 *Rd, vec3 *P, vec3 *N, vec3 *out_color) {
    vec3 light_color, color = scene->ambient_light->color;
    vec3 _l, *L = &_l;
    vec3 _h, *H = &_h;
    f32 d, d2, li, diff, spec;

    PointLight *light;
    for (u8 i = 0; i < POINT_LIGHT_COUNT; i++) {
        light = &scene->point_lights[i];
        subVec3(&light->position, P, L);

        d2 = squaredLengthVec3(L);
        d = sqrtf(d2);
        iscaleVec3(L, 1.0f / d);
        if (inShadow(scene, bvh_nodes, masks, L, P, d)) continue;

        subVec3(L, Rd, H);
        norm3(H);

        li = light->intensity / d2;
        diff = li * sdot(N, L);
        spec = li * powf(sdot(N, H), 16);

        scaleVec3(&light->color, diff + spec, &light_color);
        iaddVec3(&color, &light_color);
    }

    iaddVec3(out_color, &color);
}