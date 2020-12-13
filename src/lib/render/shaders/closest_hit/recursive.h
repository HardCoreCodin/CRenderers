#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"
#include "lib/globals/raytracing.h"
#include "../intersection/sphere.h"
#include "../intersection/plane.h"
#include "../any_hit/shadow.h"
#include "common.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void shadeRecursive(Scene *scene, Material *material, vec3 *Rd, vec3 *P, vec3 *N, bool back_facing, u8 depth, vec3 *out_color) {
    vec3 color, light_color;
    vec3 _rl, *RLd = &_rl;
    vec3 _rr, *RRd = &_rr;
    vec3 _l, *L = &_l;
    vec3 _h, *H = &_h;
    f32 d, d2, li, diff, spec, NdotRd;
    MaterialSpec mat; f32 di, si; u8 exp;
    decodeMaterial(material, mat, di, si, exp);

    if (mat.has.reflection || mat.has.refraction) fillVec3(&color, 0);
    else color = scene->ambient_light->color;

    if (mat.uses.phong || mat.has.reflection || mat.has.refraction) {
        NdotRd = -sdotInv(N, Rd);
        reflect(Rd, N, NdotRd, RLd);
    }
    if (mat.has.reflection || mat.has.refraction) {
        u8 new_hit_depth = depth + 1;
        if (new_hit_depth < MAX_HIT_DEPTH) {
            vec3 reflected_color,
                 refracted_color;
            fillVec3(&reflected_color, 0);
            fillVec3(&refracted_color, 0);

            Ray ray;
            ray.origin = P;

            if (mat.has.reflection) {
                ray.direction = RLd;
                ray.hit.uv.x = ray.hit.uv.y = 1;
                ray.hit.distance = MAX_DISTANCE;
                hitPlanes(        scene->planes, &ray);
                hitSpheresSimple(scene->spheres, &ray, 1+2+4+8);
                shadeRecursive(scene, &scene->materials[ray.hit.material_id], RRd, &ray.hit.position, &ray.hit.normal, ray.hit.is_back_facing, new_hit_depth, &reflected_color);
            }

            if (mat.has.refraction) {
                ray.direction = RRd;
                ray.hit.uv.x = ray.hit.uv.y = 1;
                ray.hit.distance = MAX_DISTANCE;
                refract(Rd, N, NdotRd, back_facing ? material->n2_over_n1 : material->n1_over_n2, RRd);
                hitPlanes(        scene->planes, &ray);
                hitSpheresSimple(scene->spheres, &ray, 1+2+4+8);
                shadeRecursive(scene, &scene->materials[ray.hit.material_id], RRd, &ray.hit.position, &ray.hit.normal, ray.hit.is_back_facing, new_hit_depth, &refracted_color);
            }

            if (mat.has.reflection && mat.has.refraction) {
                f32 reflection_amount = schlickFresnel(back_facing ? IOR_GLASS : IOR_AIR, back_facing ? IOR_AIR : IOR_GLASS, NdotRd);
                iscaleVec3(&reflected_color, reflection_amount);
                iscaleVec3(&refracted_color, 1 - reflection_amount);
            }

            if (mat.has.reflection) iaddVec3(out_color, &reflected_color);
            if (mat.has.refraction) iaddVec3(out_color, &refracted_color);
        }
    }

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