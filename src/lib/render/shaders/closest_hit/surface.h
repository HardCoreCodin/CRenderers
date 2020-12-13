#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"
#include "lib/globals/raytracing.h"
#include "common.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeSurface(Scene *scene, u8 material_id, vec3 *Rd, vec3 *P, vec3 *N, vec3 *out_color) {
    vec3 color, light_color;
    vec3 _rl, *RLd = &_rl;
    vec3 _l, *L = &_l;
    vec3 _h, *H = &_h;
    f32 NdotRd, d, d2, li, diff, spec;
    Material* material = &scene->materials[material_id];
    MaterialSpec mat; f32 di, si; u8 exp;
    decodeMaterial(material, mat, di, si, exp);

    if (mat.uses.phong || mat.has.reflection || mat.has.refraction) {
        NdotRd = -sdotInv(N, Rd);
        reflect(Rd, N, NdotRd, RLd);
    }
    if (mat.has.reflection || mat.has.refraction) fillVec3(&color, 0);
    else color = scene->ambient_light->color;

    PointLight *light;
    for (u8 i = 0; i < POINT_LIGHT_COUNT; i++) {
        light = &scene->point_lights[i];
        subVec3(&light->position, P, L);

        d2 = squaredLengthVec3(L);
        d = sqrtf(d2);
        iscaleVec3(L, 1.0f / d);
        if (inShadow(scene, L, P, d)) continue;

        if (mat.uses.blinn) {
            subVec3(L, Rd, H);
            norm3(H);
        }
        li = light->intensity / d2;
        diff = mat.has.diffuse  ? (li * di * sdot(N, L)) : 0;
        spec = mat.has.specular ? (li * si * powf(mat.uses.blinn ? sdot(N, H) : sdot(RLd, L), exp)) : 0;

        scaleVec3(&light->color, diff + spec, &light_color);
        iaddVec3(&color, &light_color);
    }

    if (mat.has.diffuse) imulVec3(&color, &material->diffuse_color);
    iaddVec3(out_color, &color);
}