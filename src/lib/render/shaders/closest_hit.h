#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/nodes/scene.h"
#include "intersection.h"

#define MAX_COLOR_VALUE 0xFF

vec3 white_color   = {1, 1, 1};
vec3 ambient_color = {10, 10, 20 };

u8 LAMBERT    = 1 << 0;
u8 PHONG      = 1 << 1;
u8 BLINN      = 1 << 2;
u8 REFLECTION = 1 << 3;
u8 REFRACTION = 1 << 4;

#define MAX_HIT_DEPTH 3
#define IOR_AIR 1
#define IOR_GLASS 1.5f
f32 n1_over_n2_for_air_and_glass = IOR_AIR / IOR_GLASS;

#define saturate(value) fmaxf( 0.0f, fminf(value, 1.0f))
inline f32 sdot(vec3* X, vec3* Y) { return saturate(dotVec3(X, Y));}
inline f32 sdotInv(vec3* X, vec3* Y) { return saturate(-dotVec3(X, Y));}

inline f32 schlickFresnel(f32 n1, f32 n2, f32 NdotL) {
    f32 R0 = (n1 - n2) / (n1 + n2);
    return R0 + (1 - R0)*powf(1 - NdotL, 5);
}
inline void reflect(vec3 *V, vec3 *N, f32 NdotV, vec3 *R) {
    vec3 NdotV2N;
    scaleVec3(N, 2 * NdotV, &NdotV2N);
    addVec3(V, &NdotV2N, R);
}

inline void refract(vec3* V, vec3* N, f32 NdotV, f32 n1_over_n2, vec3* out) {
//    if (NdotV == -1) NdotV = dotVec3(N, V);
    f32 c = n1_over_n2*n1_over_n2 * (1 - (NdotV*NdotV));
    if (c >= 1) {
        reflect(V, N, NdotV, out);
        return;
    }
//    if (1 - c < EPS) c = 0;
    c = sqrtf(1 - c);
//    if (NdotV <= 0) c = -c;
    vec3 a, b;
    scaleVec3(V, n1_over_n2, &a);
    scaleVec3(N, n1_over_n2 * -NdotV - c, &b);
    addVec3(&a, &b, out);
    norm3(out);
}

inline void shadeNormal(vec3* normal, f32 distance, vec3* out_color) {
    vec3 normal_color;
    addVec3(normal, &white_color, &normal_color);
    scaleVec3(&normal_color, 4 * MAX_COLOR_VALUE / distance, out_color);
}

inline void shadeDirection(vec3* direction, vec3* out_color) {
    vec3 normal_color;
    addVec3(direction, &white_color, &normal_color);
    scaleVec3(&normal_color, 0.5f * MAX_COLOR_VALUE, out_color);
}

inline f32 shadeDiffuse(HitInfo *hit_info, f32 intensity) {
//    if (!hit_info->has_NdotL) {
//        hit_info->has_NdotL = true;
//        hit_info->NdotH = sdot(hit_info->N, &hit_info->L);
//    }

    return hit_info->NdotL * intensity;
}

#define computeSpecularity(intensity, exp, RLdotL) intensity * powf(RLdotL, exp)

inline f32 shadePhong(HitInfo *hit_info, f32 intensity, u8 exponent) {
//    if (!hit_info->has_RLdotL) {
//        hit_info->has_RLdotL = true;
//        if (!hit_info->has_RL) {
//            hit_info->has_RL = true;
//            if (!hit_info->has_NdotV) {
//                hit_info->has_NdotV = true;
//                hit_info->NdotV = -saturate(-dotVec3(hit_info->N, hit_info->V));
//            }
//            reflect(hit_info->V, hit_info->N, hit_info->NdotV, &hit_info->RL);
//        }
//        hit_info->RLdotL = saturate(dotVec3(&hit_info->RL, &hit_info->L));
//    }

    return computeSpecularity(intensity, exponent, hit_info->RLdotL);
}

inline f32 shadeBlinn(HitInfo *hit_info, f32 intensity, u8 exponent) {
//    if (!hit_info->has_NdotH) {
//        hit_info->has_NdotH = true;
//        if (!hit_info->has_H) {
//            hit_info->has_H = true;
//            subVec3(&hit_info->L, hit_info->V, &hit_info->H);
//            norm3(&hit_info->H);
//        }
//        hit_info->NdotH = sdot(hit_info->N, &hit_info->H);
//    }

    return computeSpecularity(intensity, exponent, hit_info->NdotH);
}

inline void shade(RayHit *hit, Material* material, vec3 *out_color);

inline bool shadeReflection(RayHit *hit, HitInfo *hit_info, vec3* reflected_color) {
    if (hit->hit_depth + 1 > MAX_HIT_DEPTH) return false;

//    if (!hit_info->has_RL) {
//        hit_info->has_RL = true;
//        if (!hit_info->has_NdotV) {
//            hit_info->has_NdotV = true;
//            hit_info->NdotV = -saturate(-dotVec3(hit_info->N, hit_info->V));
//        }
//        reflect(hit_info->V, hit_info->N, hit_info->NdotV, &hit_info->RL);
//    }

    RayHit reflection_hit;
    reflection_hit.distance = MAX_DISTANCE;
    reflection_hit.hit_depth = hit->hit_depth + 1;
    reflection_hit.ray_origin = hit->position;
    reflection_hit.ray_direction = hit_info->RL;

    Material* material;
    hitPlanes(&reflection_hit, &material);
    hitSpheres(&reflection_hit, &material, false, NULL);

    shade(&reflection_hit, material, reflected_color);

    return true;
}

inline bool shadeRefraction(RayHit *hit, HitInfo *hit_info, vec3* refracted_color) {
    if (hit->hit_depth + 1 > MAX_HIT_DEPTH) return false;

//    if (!hit_info->has_NdotV) {
//        hit_info->has_NdotV = true;
//        hit_info->NdotV = sdotInv(hit_info->N, hit_info->V);
//    }

    RayHit refraction_hit;
    refraction_hit.distance = MAX_DISTANCE;
    refraction_hit.hit_depth = hit->hit_depth + 1;
    refraction_hit.ray_origin = hit->position;

    refract(&hit->ray_direction, &hit->normal, hit_info->NdotV, hit->inner ? hit->n2_over_n1 : hit->n1_over_n2, &refraction_hit.ray_direction);
    Material* material;
    hitPlanes(&refraction_hit, &material);
    hitSpheres(&refraction_hit, &material, false, NULL);
    shade(&refraction_hit, material, refracted_color);

    return true;
}

inline bool inShadow(vec3* Rd, vec3* Ro, f32 light_distance) {
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    Sphere* sphere;
    f32 t, dt;
    vec3 _I, *I = &_I,
            _C, *C = &_C;

    for (sphere = scene.spheres; sphere != last_sphere; sphere++) {
        subVec3(&sphere->position, Ro, C);
        t = dotVec3(C, Rd);
        if (t > 0 && t < light_distance) {
            scaleVec3(Rd, t, I);
            isubVec3(I, C);
            dt = sphere->radius*sphere->radius - squaredLengthVec3(I);
            if (dt > 0 && t > sqrt(dt)) return true;
        }
    }

    return false;
}

inline void shade(RayHit *hit, Material* material, vec3 *out_color) {
    f32 li,
        light_distance_squared,
        light_distance,
        diffuse_light,
        specular_light;

//    *out_color = ambient_color;
    HitInfo hit_info;
//    hit_info.has_NdotV = hit_info.has_RL = hit_info.had_RR = true;
    hit_info.N = &hit->normal;
    hit_info.V = &hit->ray_direction;
    vec3* P = &hit->position;
    vec3* L = &hit_info.L;
    vec3* N = hit_info.N;
    vec3* V = hit_info.V;
    vec3* H = &hit_info.H;
    vec3* RL = &hit_info.RL;

    bool using_phong = material->specular_shader == shadePhong;
    bool using_blinn = material->specular_shader == shadeBlinn;
    bool has_diffuse = material->diffuse_shader == shadeDiffuse;
    bool has_specular = using_phong || using_blinn;
    bool has_reflection = material->has_reflection;
    bool has_refraction = material->has_refraction;
    bool has_rfl_or_rfr = has_reflection || has_refraction;

    f32 di = material->diffuse_intensity;
    f32 si = material->specular_intensity;
    f32 exp = material->specular_exponent * (using_blinn ? 4 : 1);

    if (using_phong || has_rfl_or_rfr) hit_info.NdotV = sdotInv(N, V);
    if (using_phong || has_reflection) reflect(V, N, hit_info.NdotV, RL);

    vec3 accumulated_color, accumulated_light_color, scaled_light_color;
    fillVec3(&accumulated_color, 0);
    fillVec3(&accumulated_light_color, 0);

    if (has_rfl_or_rfr) {
        u8 new_hit_depth = hit->hit_depth + 1;
        if (new_hit_depth < MAX_HIT_DEPTH) {
            vec3 reflected_color,
                 refracted_color;
            fillVec3(&reflected_color, 0);
            fillVec3(&refracted_color, 0);

            Material* secondary_hit_material;
            RayHit secondary_hit;
            secondary_hit.hit_depth = new_hit_depth;
            secondary_hit.ray_origin = *P;

            if (has_reflection) {
                secondary_hit.distance = MAX_DISTANCE;
                secondary_hit.ray_direction = *RL;
                hitPlanes(&secondary_hit, &secondary_hit_material);
                hitSpheres(&secondary_hit, &secondary_hit_material, false, NULL);
                shade(&secondary_hit, secondary_hit_material, &reflected_color);
            }

            if (has_refraction) {
                secondary_hit.distance = MAX_DISTANCE;
                refract(&hit->ray_direction, N, hit_info.NdotV, hit->inner ? hit->n2_over_n1 : hit->n1_over_n2, &secondary_hit.ray_direction);
                hitPlanes(&secondary_hit, &secondary_hit_material);
                hitSpheres(&secondary_hit, &secondary_hit_material, false, NULL);
                shade(&secondary_hit, secondary_hit_material, &refracted_color);
            }

            if (has_reflection && has_refraction) {
                f32 reflection_amount = schlickFresnel(hit->inner ? IOR_GLASS : IOR_AIR, hit->inner ? IOR_AIR : IOR_GLASS, hit_info.NdotV);
                iscaleVec3(&reflected_color, reflection_amount);
                iscaleVec3(&refracted_color, 1 - reflection_amount);
            }

            if (has_reflection) iaddVec3(out_color, &reflected_color);
            if (has_refraction) iaddVec3(out_color, &refracted_color);
        }
    }
//    bool has_reflection = material->reflection_shader && material->reflection_shader(hit, &hit_info, &reflected_color);
//    bool has_refraction = material->refraction_shader && material->refraction_shader(hit, &hit_info, &refracted_color);
//
//    if (has_reflection &&
//        has_refraction) {
//        if (!hit_info.has_NdotV) {
//            hit_info.has_NdotV = true;
//            hit_info.NdotV = sdotInv(hit_info.N, hit_info.V);
//        }
//        f32 reflection_amount = schlickFresnel(hit->inner ? IOR_GLASS : IOR_AIR, hit->inner ? IOR_AIR : IOR_GLASS, hit_info.NdotV);
//        iscaleVec3(&reflected_color, reflection_amount);
//        iscaleVec3(&refracted_color, 1 - reflection_amount);
//
//    }
//    if (has_reflection) iaddVec3(out_color, &reflected_color);
//    if (has_refraction) iaddVec3(out_color, &refracted_color);

    PointLight *last_light = scene.point_lights + scene.light_count;
    for (PointLight* light = scene.point_lights; light != last_light; light++) {
        subVec3(&light->position, P, L);
        light_distance_squared = squaredLengthVec3(L);
        light_distance = sqrtf(light_distance_squared);
        iscaleVec3(L, 1.0f / light_distance);
        if (inShadow(L, P, light_distance)) continue;

        if (using_blinn) {
            subVec3(L, V, H);
            norm3(H);
        }
        li = light->intensity / light_distance_squared;
        diffuse_light  = has_diffuse ? (li * di * sdot(N, L)) : 0;
        specular_light = has_specular ? (li * si * powf(using_blinn ? sdot(N, H) : sdot(RL, L), exp)) : 0;

        scaleVec3(&light->color, diffuse_light + specular_light, &scaled_light_color);
        iaddVec3(out_color, &scaled_light_color);
    }

//    f32 r = material->diffuse_color.x;
//    f32 g = material->diffuse_color.y;
//    f32 b = material->diffuse_color.z;
//    if (r != 1.0f ||
//        g != 1.0f ||
//        b != 1.0f) {
//        accumulated_color.x = accumulated_light_color.x * r;
//        accumulated_color.y = accumulated_light_color.y * r;
//        accumulated_color.z = accumulated_light_color.z * r;
//        iaddVec3(out_color, &accumulated_color);
//    } else iaddVec3(out_color, &accumulated_light_color);
//    iaddVec3(out_color, &accumulated_light_color);
}

void initShaders() {
    Material* walls_material = scene.materials;
    Material* diffuse_ball_material = scene.materials + 1;
    Material* specular_ball_material_phong = scene.materials + 2;
    Material* specular_ball_material_blinn = scene.materials + 3;
    Material* reflective_ball_material = scene.materials + 4;
    Material* refractive_ball_material = scene.materials + 5;
    Material* reflective_refractive_ball_material = scene.materials + 6;

    fillVec3(&walls_material->diffuse_color, 1);
    walls_material->uses = LAMBERT;
    walls_material->has_diffuse = true;
    walls_material->has_specular = false;
    walls_material->has_reflection = false;
    walls_material->has_refraction = false;
    walls_material->has_transparency = false;
    walls_material->specular_intensity = 0;
    walls_material->diffuse_intensity = 1;
    walls_material->specular_exponent = 0;
    walls_material->diffuse_shader= shadeDiffuse;
    walls_material->specular_shader= NULL;
    walls_material->reflection_shader= NULL;
    walls_material->refraction_shader = NULL;

    fillVec3(&diffuse_ball_material->diffuse_color, 1);
    diffuse_ball_material->uses = LAMBERT;
    diffuse_ball_material->has_diffuse = true;
    diffuse_ball_material->has_specular = false;
    diffuse_ball_material->has_reflection = false;
    diffuse_ball_material->has_refraction = false;
    diffuse_ball_material->has_transparency = false;
    diffuse_ball_material->specular_intensity = 0;
    diffuse_ball_material->diffuse_intensity = 1;
    diffuse_ball_material->specular_exponent = 0;
    diffuse_ball_material->diffuse_shader= shadeDiffuse;
    diffuse_ball_material->specular_shader= NULL;
    diffuse_ball_material->reflection_shader= NULL;
    diffuse_ball_material->refraction_shader = NULL;

    fillVec3(&specular_ball_material_phong->diffuse_color, 1);
    specular_ball_material_phong->uses = LAMBERT | PHONG;
    specular_ball_material_phong->has_diffuse = true;
    specular_ball_material_phong->has_specular = true;
    specular_ball_material_phong->has_reflection = false;
    specular_ball_material_phong->has_refraction = false;
    specular_ball_material_phong->has_transparency = false;
    specular_ball_material_phong->specular_intensity = 1;
    specular_ball_material_phong->diffuse_intensity = 1;
    specular_ball_material_phong->specular_exponent = 4;
    specular_ball_material_phong->diffuse_shader= shadeDiffuse;
    specular_ball_material_phong->specular_shader= shadePhong;
    specular_ball_material_phong->reflection_shader= NULL;
    specular_ball_material_phong->refraction_shader = NULL;

    fillVec3(&specular_ball_material_blinn->diffuse_color, 1);
    specular_ball_material_blinn->uses = LAMBERT | BLINN;
    specular_ball_material_blinn->has_diffuse = true;
    specular_ball_material_blinn->has_specular = true;
    specular_ball_material_blinn->has_reflection = false;
    specular_ball_material_blinn->has_refraction = false;
    specular_ball_material_blinn->has_transparency = false;
    specular_ball_material_blinn->specular_intensity = 1;
    specular_ball_material_blinn->diffuse_intensity = 1;
    specular_ball_material_blinn->specular_exponent = 4;
    specular_ball_material_blinn->diffuse_shader= shadeDiffuse;
    specular_ball_material_blinn->specular_shader= shadeBlinn;
    specular_ball_material_blinn->reflection_shader= NULL;
    specular_ball_material_blinn->refraction_shader = NULL;

    fillVec3(&reflective_ball_material->diffuse_color, 1);

    reflective_ball_material->uses = BLINN | REFLECTION;
    reflective_ball_material->has_diffuse = false;
    reflective_ball_material->has_specular = true;
    reflective_ball_material->has_reflection = true;
    reflective_ball_material->has_refraction = false;
    reflective_ball_material->has_transparency = false;
    reflective_ball_material->specular_intensity = 1;
    reflective_ball_material->diffuse_intensity = 0;
    reflective_ball_material->specular_exponent = 4;
    reflective_ball_material->diffuse_shader= NULL;
    reflective_ball_material->specular_shader= shadeBlinn;
    reflective_ball_material->reflection_shader= shadeReflection;
    reflective_ball_material->refraction_shader = NULL;

    fillVec3(&refractive_ball_material->diffuse_color, 1);

    refractive_ball_material->uses = BLINN | REFRACTION;
    refractive_ball_material->has_diffuse = false;
    refractive_ball_material->has_specular = true;
    refractive_ball_material->has_reflection = false;
    refractive_ball_material->has_refraction = true;
    refractive_ball_material->has_transparency = false;
    refractive_ball_material->specular_intensity = 1;
    refractive_ball_material->diffuse_intensity = 0;
    refractive_ball_material->specular_exponent = 4;
    refractive_ball_material->diffuse_shader= NULL;
    refractive_ball_material->specular_shader= shadeBlinn;
    refractive_ball_material->reflection_shader= NULL;
    refractive_ball_material->refraction_shader = shadeRefraction;

    fillVec3(&reflective_refractive_ball_material->diffuse_color, 1);
    reflective_refractive_ball_material->uses = BLINN | REFLECTION | REFRACTION;
    reflective_refractive_ball_material->has_diffuse = false;
    reflective_refractive_ball_material->has_specular = true;
    reflective_refractive_ball_material->has_reflection = true;
    reflective_refractive_ball_material->has_refraction = true;
    reflective_refractive_ball_material->has_transparency = false;
    reflective_refractive_ball_material->specular_intensity = 1;
    reflective_refractive_ball_material->diffuse_intensity = 0;
    reflective_refractive_ball_material->specular_exponent = 4;
    reflective_refractive_ball_material->diffuse_shader= NULL;
    reflective_refractive_ball_material->specular_shader= shadeBlinn;
    reflective_refractive_ball_material->reflection_shader= shadeReflection;
    reflective_refractive_ball_material->refraction_shader = shadeRefraction;

//    specular_ball_material_phong->diffuse_color.x = 0.5;
//    specular_ball_material_blinn->diffuse_color.y = 0.5;
//    diffuse_ball_material->diffuse_color.z = 0.5;
//    diffuse_ball_material->diffuse_color.z = 0.5;
}

//
//inline void shadeRefractionDoubleSided(RayHit* hit, RayHit* farther_hit, vec3* out_color) {
//    f32 light_intensity,
//            light_attenuation,
//            light_distance,
//            RdotL,
//            N1dotV,
//            N2dotV;
//    vec3 L, RL1, RL2, RR1, RR2, scaled_light_color;
//    vec3 *V = &hit->ray_direction;
//    vec3 *N1 = &hit->normal;
//    vec3 *P1 = &hit->position;
//    vec3 *N2 = &farther_hit->normal;
//    vec3 *P2 = &farther_hit->position;
//
//    f32 fresnel = 0;
//    vec3 reflected_color;
//
//    if (hit->distance) {
//        N1dotV = dotVec3(N1, V);
//        reflect(V, N1, N1dotV, &RL1);
//        refract(V, N1, N1dotV, n1_over_n2_for_air_and_glass, &RR1);
//
//        N2dotV = dotVec3(N2, &RR1);
//        reflect(&RR1, N2, N2dotV, &RL2);
//        refract(&RR1, N2, N2dotV, IOR_GLASS, &RR2);
//
//        reflected_color = ambient_color;
//        PointLight *last_light = scene.point_lights + scene.light_count;
//        for (PointLight* light = scene.point_lights; light != last_light; light++) {
//            subVec3(&light->position, P1, &L);
//            light_distance = squaredLengthVec3(&L);
//            light_attenuation = 1.0f / light_distance;
//            light_distance = sqrtf(light_distance);
//            iscaleVec3(&L, 1.0f / light_distance);
//            if (inShadow(&L, P1, light_distance)) continue;
//
//            RdotL = max(0.0f, min(dotVec3(&RL1, &L), 1.0f));
//
//            light_intensity = light->intensity * light_attenuation * powf(RdotL, 4);
//
//            scaleVec3(&light->color, light_intensity, &scaled_light_color);
//            iaddVec3(&reflected_color, &scaled_light_color);
//        }
//
////        N1dotV = max(0.0f, min(-N1dotV, 1.0f));
////        fresnel = max(0.0f, min(schlickFresnel(IOR_GLASS, IOR_AIR, N1dotV), 1.0f));
////        iscaleVec3(&reflected_color, fresnel);
//    } else {
//        N2dotV = dotVec3(N2, V);
//        reflect(V, N2, N2dotV, &RL2);
//        refract(V, N2, N2dotV, IOR_GLASS, &RR2);
//    }
//
////    iscaleVec3(&RR2, -1);
//    refraction_hit.ray_direction = RR2;
//    refraction_hit.ray_origin = *P2;
//    hitPlanes(&refraction_hit);
//    hitSpheres(&refraction_hit, false);
//
//    shaders[refraction_hit.material.shader_id](&refraction_hit, out_color);
//
////    if (hit->distance) {
////        iscaleVec3(out_color, 1 - fresnel);
////        iaddVec3(out_color, &reflected_color);
////    }
//}
//
//inline void shadeReflectionRefractionDoubleSided(RayHit* hit, RayHit* farther_hit, vec3* out_color) {
//    f32 light_intensity,
//            light_attenuation,
//            light_distance,
//            RdotL,
//            N1dotV,
//            N2dotV;
//    vec3 L, RL1, RL2, RR1, RR2, scaled_light_color;
//    vec3 *V = &hit->ray_direction;
//    vec3 *N1 = &hit->normal;
//    vec3 *P1 = &hit->position;
//    vec3 *N2 = &farther_hit->normal;
//    vec3 *P2 = &farther_hit->position;
//
//    f32 fresnel = 0;
//    vec3 reflected_color;
//
//    if (hit->distance) {
//        N1dotV = dotVec3(N1, V);
//        reflect(V, N1, N1dotV, &RL1);
//        refract(V, N1, N1dotV, n1_over_n2_for_air_and_glass, &RR1);
//
//        N2dotV = dotVec3(N2, &RR1);
//        reflect(&RR1, N2, N2dotV, &RL2);
//        refract(&RR1, N2, N2dotV, IOR_GLASS, &RR2);
//
//        reflected_color = ambient_color;
//
//        vec3 L, scaled_light_color;
//        vec3 *V = &hit->ray_direction;
//        vec3 *N = &hit->normal;
//        vec3 *P = &hit->position;
//
//        refract(V, N, dotVec3(N, V), n1_over_n2_for_air_and_glass, &refraction_hit.ray_direction);
//
//        refraction_hit.ray_origin = *P;
//        hitPlanes(&refraction_hit);
//        hitSpheres(&refraction_hit, false);
//
//        reflection_hit.ray_origin    = *P1;
//        reflection_hit.ray_direction = RL1;
//        hitPlanes(&reflection_hit);
//        hitSpheres(&reflection_hit, false);
//        shaders[reflection_hit.material.shader_id](&reflection_hit, out_color);
//
//        PointLight *last_light = scene.point_lights + scene.light_count;
//        for (PointLight* light = scene.point_lights; light != last_light; light++) {
//            subVec3(&light->position, P1, &L);
//            light_distance = squaredLengthVec3(&L);
//            light_attenuation = 1.0f / light_distance;
//            light_distance = sqrtf(light_distance);
//            iscaleVec3(&L, 1.0f / light_distance);
//            if (inShadow(&L, P1, light_distance)) continue;
//
//            RdotL = max(0.0f, min(dotVec3(&RL1, &L), 1.0f));
//
//            light_intensity = light->intensity * light_attenuation * powf(RdotL, 4);
//
//            scaleVec3(&light->color, light_intensity, &scaled_light_color);
//            iaddVec3(out_color, &scaled_light_color);
//        }
//
////        N1dotV = min(-1.0f, max(N1dotV, 0.0f));
//        fresnel = max(0.0f, min(schlickFresnel(IOR_GLASS, IOR_AIR, -N1dotV), 1.0f));
//        iscaleVec3(&reflected_color, fresnel);
//    } else {
//        N2dotV = dotVec3(N2, V);
//        reflect(V, N2, N2dotV, &RL2);
//        refract(V, N2, N2dotV, IOR_GLASS, &RR2);
//    }
//
//    refraction_hit.ray_origin    = *P2;
//    refraction_hit.ray_direction = RR2;
//    hitPlanes(&refraction_hit);
//    hitSpheres(&refraction_hit, false);
//    shaders[refraction_hit.material.shader_id](&refraction_hit, out_color);
//
//    if (hit->distance) {
//        iscaleVec3(out_color, 1 - fresnel);
//        iaddVec3(out_color, &reflected_color);
//    }
//}


//
//
//
//
//
//void refract8(vec3* Rd, vec3* N, vec3* R, f32 eta) {
//    vec3 I;
//    scaleVec3(Rd, -1, &I);
//    f32 NdotI = dotVec3(N, &I);
//    bool backside = NdotI < 0.0f;
//    f32 rcp = backside ? 1.0f / eta : eta;
//    if ((rcp * rcp * (NdotI * NdotI - 1) + 1.0f) < 0) {
//        rcp++;
//    }
//    f32 abs_NdotT = sqrtf(rcp * rcp * (NdotI * NdotI - 1) + 1.0f);
//    vec3 a, b;
//    scaleVec3(N, NdotI * rcp + (backside ? abs_NdotT : -abs_NdotT), &a);
//    scaleVec3(&I, rcp, &b);
//    addVec3(&a, &b, R);
//}
//
//inline void refract7(vec3 *Rd, vec3 *N, vec3 *RRd, f32 n1_over_n2) {
//    // Construct an ortho-normal basis for a new space where N becomes Y pointing downwards:
//    mat3 ortho_normal_basis;
//    ortho_normal_basis.X.x = -N->x;
//    ortho_normal_basis.X.y = -N->y;
//    ortho_normal_basis.X.z = -N->z;
//    vec3 *invN = &ortho_normal_basis.X;
//    crossVec3(Rd, invN, &ortho_normal_basis.Z);
//    f32 sin_alpha = sqrtf(squaredLengthVec3(&ortho_normal_basis.Z));
//    iscaleVec3(&ortho_normal_basis.Z, 1/sin_alpha);
//
//    crossVec3(invN, &ortho_normal_basis.Z, &ortho_normal_basis.Y);
//    norm3(&ortho_normal_basis.Y);
//
//    f32 sin_beta = sin_alpha * n1_over_n2; // Snell's law
//
//    mat3 rotation_matrix;
//    setMat3ToIdentity(&rotation_matrix);
//    rotation_matrix.X.x = rotation_matrix.Y.y = cosf(asinf(sin_beta));
//    rotation_matrix.X.y = -sin_beta;
//    rotation_matrix.Y.x = +sin_beta;
//
//    mat3 inv_ortho_normal_basis;
//    transposeMat3(&ortho_normal_basis, &inv_ortho_normal_basis);
//    mulVec3Mat3(Rd, &inv_ortho_normal_basis, RRd); // Re-represent the ray direction in the new space
//    imulVec3Mat3(RRd, &rotation_matrix);
//    imulVec3Mat3(RRd, &ortho_normal_basis);
//}
//inline void refract5(vec3 *Rd, vec3 *N, vec3 *RRd, f32 n1_over_n2) {
//    // Construct an ortho-normal basis for a new space where N becomes Y pointing downwards:
//    mat3 ortho_normal_basis;
//    ortho_normal_basis.X.x = -N->x;
//    ortho_normal_basis.X.y = -N->y;
//    ortho_normal_basis.X.z = -N->z;
//    vec3* invN = &ortho_normal_basis.X;
//
//    //  Rd • invN = cos(a)
//    // RRd • invN = cos(b)
//    // sin(a)n1 = sin(b)n2
//    // sin(b) = (n1/n2)sin(a)
//    // sin(a) = | Rd ⊕ invN|
//    // sin(b) = |RRd ⊕ invN|
//    // |RRd ⊕ invN| = (n1/n2)|Rd ⊕ invN|
//    // √((RRd ⊕ invN) • (RRd ⊕ invN)) = (n1/n2)√((Rd ⊕ invN) • (Rd ⊕ invN))
//    // ((RRd ⊕ invN) • (RRd ⊕ invN)) = (n1/n2)²((Rd ⊕ invN) • (Rd ⊕ invN))
//    // RRd.x
//
//    // |RRd ⊕ invN| =
//    // n1/n2 = |Rd ⊕ invN| /
//
//    vec3 invNcrossRd;
//    crossVec3(invN, Rd, &invNcrossRd);
//
////    f32 cos_alpha = dotVec3(Rd, invN);
////    f32 alpha = acosf(cos_alpha);
////    f32 sin_alpha = sinf(alpha);
//    f32 sin_alpha = sqrtf(squaredLengthVec3(&invNcrossRd));
//    f32 sin_beta = sin_alpha * n1_over_n2; // Snell's law
//    f32 beta  = asinf(sin_beta);
//
//    crossVec3(sin_alpha > 0 ? invN : N, Rd, &ortho_normal_basis.Z);
//    crossVec3(&ortho_normal_basis.Z, invN, &ortho_normal_basis.X);
//
//    mat3 rotation_matrix;
//    setMat3ToIdentity(&rotation_matrix);
//    rotation_matrix.X.x = rotation_matrix.Y.y = cosf(beta);
//    rotation_matrix.X.y = -sin_beta;
//    rotation_matrix.Y.x = +sin_beta;
//
//    mat3 inv_ortho_normal_basis;
//    transposeMat3(&ortho_normal_basis, &inv_ortho_normal_basis);
//    mulVec3Mat3(Rd, &inv_ortho_normal_basis, RRd); // Re-represent the ray direction in the new space
//    imulVec3Mat3(RRd, &rotation_matrix);
//    imulVec3Mat3(RRd, &ortho_normal_basis);
//}
//
//
//inline void refract4(vec3 *Rd, vec3 *N, vec3 *RRd, f32 n1_over_n2) {
//    f32 a, b, y;
//    vec3 X, Z, bX, yN;
//    crossVec3(N, Rd, &Z);  // Z = N ⊕ Rd
//    crossVec3(&Z, N, &X);  // X = Z ⊕ N
//    norm3(&Z);
//    norm3(&X);
//    a = dotVec3(Rd, &X);   // a = Rd • X
//    b = n1_over_n2 * a;    // b = a•(n1/n2)
//    if (-dotVec3(Rd, N) < 0.0f) b -= 1;
//    y = sqrtf(1 - b*b); // y = √(1² - b²)
//    scaleVec3(&X, b, &bX);  // bX = b•X
//    scaleVec3(N, y, &yN);   // yN = y•N
//    subVec3(&bX, &yN, RRd); // RRd = bX - yN
//}
//inline void refract6(vec3 *Rd, vec3 *N, vec3 *RRd, f32 n1_over_n2) {
//    f32 a, b, b2, y;
//    vec3 X, Z, bX, yN;
//    bool backside = -dotVec3(Rd, N) < 0.0f;
//
//    crossVec3(N, Rd, &Z);  // Z = N ⊕ Rd
//    crossVec3(&Z, N, &X);  // X = Z ⊕ N
//    norm3(&Z);
//    norm3(&X);
//    a = dotVec3(Rd, &X);   // a = Rd • X
//    b = n1_over_n2 * a;    // b = a•(n1/n2)
////    if (backside) b = 1/b;
//    b2 = b * b;
//    if (b2 > 1) b2--;
//    y = sqrtf(1 - b2);
//    scaleVec3(&X, b, &bX);  // bX = b•X
//    scaleVec3(N, y, &yN);   // yN = y•N
//    subVec3(&bX, &yN, RRd); // RRd = bX - yN
//}