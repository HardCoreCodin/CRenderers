#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/nodes/scene.h"
#include "intersection.h"

#define MAX_COLOR_VALUE 0xFF

vec3 white_color   = {1, 1, 1};
vec3 ambient_color = {20, 20, 40 };

#define MAX_HIT_DEPTH 4
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
    scaleVec3(N, -2 * NdotV, &NdotV2N);
    addVec3(V, &NdotV2N, R);
}

inline void refract(vec3* V, vec3* N, f32 NdotV, f32 n1_over_n2, vec3* out) {
    f32 c = n1_over_n2*n1_over_n2 * (1 - (NdotV*NdotV));
    if (c + EPS > 1) {
        reflect(V, N, NdotV, out);
        return;
    }
    c = sqrtf(1 - c);
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

RayHit shadow_ray_hit;
inline bool inShadow(vec3* Rd, vec3* Ro, f32 light_distance) {
    Material* material;
    shadow_ray_hit.ray_origin = *Ro;
    shadow_ray_hit.ray_direction = *Rd;
    shadow_ray_hit.distance = light_distance;
    return hitSpheres(&shadow_ray_hit, &material, false, NULL) && shadow_ray_hit.distance < light_distance;
}
inline bool inShadowSimple(vec3 *Rd, vec3* Ro, f32 light_distance) {
    vec3 hit_position, hit_position_tangent;
    f32 hit_distance;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* last_tetrahedron = scene.tetrahedra + scene.tetrahedron_count;
    for (Tetrahedron* tetrahedron = scene.tetrahedra; tetrahedron != last_tetrahedron; tetrahedron++) {
        Triangle *last_triangle = tetrahedron->triangles + 4;
        for (Triangle *triangle = tetrahedron->triangles; triangle != last_triangle; triangle++) {
            if (hitPlane(triangle->p1, triangle->normal, Rd, Ro, &hit_distance) && hit_distance < light_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                subVec3(&hit_position, triangle->p1, &hit_position_tangent);
                imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);

                if (hit_position_tangent.y > 0 &&
                    hit_position_tangent.y < SQRT_OF_THREE * hit_position_tangent.x &&
                    hit_position_tangent.y < SQRT_OF_THREE * (1 - hit_position_tangent.x))
                    return true;
            }
        }
    }

    f32 t, dt;
    vec3 _I, *I = &_I,
            _C, *C = &_C;

    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++) {
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
inline void shadeLambert(RayHit* hit, vec3* color) {
    f32 light_intensity,
            light_attenuation,
            light_distance,
            NdotL;
    vec3 L, scaled_light_color;
    vec3 *N = &hit->normal;
    vec3 *P = &hit->position;

    *color = ambient_color;

    PointLight *last_light = scene.point_lights + scene.light_count;
    for (PointLight* light = scene.point_lights; light != last_light; light++) {
        subVec3(&light->position, P, &L);
        light_distance = squaredLengthVec3(&L);
        light_attenuation = 1.0f / light_distance;
        light_distance = sqrtf(light_distance);
        iscaleVec3(&L, 1.0f / light_distance);
        if (inShadowSimple(&L, P, light_distance)) continue;

        NdotL = max(0.0f, min(dotVec3(N, &L), 1.0f));
        light_intensity = light->intensity * light_attenuation * NdotL;

        scaleVec3(&light->color, light_intensity, &scaled_light_color);
        iaddVec3(color, &scaled_light_color);
    }
}

inline void shade(RayHit *hit, Material* material, vec3 *out_color) {
    f32 li, NdotV,
        light_distance_squared,
        light_distance,
        diffuse_light,
        specular_light;
    vec3 L, H, R, color, scaled_light_color,
         *N = &hit->normal,
         *V = &hit->ray_direction,
         *P = &hit->position;

    u8 uses = material->uses;
    bool using_phong = uses & PHONG;
    bool using_blinn = uses & BLINN;
    bool has_diffuse = uses & LAMBERT;
    bool has_specular = using_phong || using_blinn;
    bool has_reflection = uses & REFLECTION;
    bool has_refraction = uses & REFRACTION;
    bool has_rfl_or_rfr = has_reflection || has_refraction;

    if (has_rfl_or_rfr) fillVec3(out_color, 0);
    else *out_color = ambient_color;

    f32 di = material->diffuse_intensity;
    f32 si = material->specular_intensity;
    u8 exp = material->specular_exponent * (using_blinn ? (u8)4 : (u8)1);

    bool from_behind;
    if (using_phong || has_rfl_or_rfr) {
        NdotV = dotVec3(N, V);
        from_behind = NdotV > 0;
        NdotV = -saturate(from_behind ? NdotV : -NdotV);
        if (from_behind)
            iscaleVec3(N, -1);
        reflect(V, N, NdotV, &R);
    }

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
                secondary_hit.ray_direction = R;
                hitPlanes(&secondary_hit, &secondary_hit_material);
                hitSpheres(&secondary_hit, &secondary_hit_material, false, NULL);
                shade(&secondary_hit, secondary_hit_material, &reflected_color);
            }

            if (has_refraction) {
                secondary_hit.distance = MAX_DISTANCE;
                refract(&hit->ray_direction, N, NdotV, from_behind ? IOR_GLASS : n1_over_n2_for_air_and_glass, &secondary_hit.ray_direction);
                hitPlanes(&secondary_hit, &secondary_hit_material);
                hitSpheres(&secondary_hit, &secondary_hit_material, false, NULL);
                shade(&secondary_hit, secondary_hit_material, &refracted_color);
            }

            if (has_reflection && has_refraction) {
                f32 reflection_amount = schlickFresnel(from_behind ? IOR_GLASS : IOR_AIR, from_behind ? IOR_AIR : IOR_GLASS, NdotV);
                iscaleVec3(&reflected_color, reflection_amount);
                iscaleVec3(&refracted_color, 1 - reflection_amount);
            }

            if (has_reflection) iaddVec3(out_color, &reflected_color);
            if (has_refraction) iaddVec3(out_color, &refracted_color);
        }
    }

    fillVec3(&color, 0);
    PointLight *last_light = scene.point_lights + scene.light_count;
    for (PointLight* light = scene.point_lights; light != last_light; light++) {
        subVec3(&light->position, P, &L);
        light_distance_squared = squaredLengthVec3(&L);
        light_distance = sqrtf(light_distance_squared);
        iscaleVec3(&L, 1.0f / light_distance);
        if (inShadow(&L, P, light_distance)) continue;

        if (using_blinn) {
            subVec3(&L, V, &H);
            norm3(&H);
        }
        li = light->intensity / light_distance_squared;
        diffuse_light  = has_diffuse  ? (li * di * sdot(N, &L)) : 0;
        specular_light = has_specular ? (li * si * powf(using_blinn ? sdot(N, &H) : sdot(&R, &L), exp)) : 0;

        scaleVec3(&light->color, diffuse_light + specular_light, &scaled_light_color);
        iaddVec3(&color, &scaled_light_color);
    }

    imulVec3(&color, &material->diffuse_color);
    iaddVec3(out_color, &color);
}

void initShaders() {
    Material *walls_material = scene.materials,
             *diffuse_ball_material = scene.materials + 1,
             *specular_ball_material_phong = scene.materials + 2,
             *specular_ball_material_blinn = scene.materials + 3,
             *reflective_ball_material = scene.materials + 4,
             *refractive_ball_material = scene.materials + 5,
             *reflective_refractive_ball_material = scene.materials + 6;

    Material* material = scene.materials;
    for (int i = 0; i < MATERIAL_COUNT; i++, material++) {
        fillVec3(&material->diffuse_color, 1);
        material->diffuse_intensity = 1;
        material->specular_intensity = 1;
        material->specular_exponent = 4;
    }

    walls_material->uses = LAMBERT;
    diffuse_ball_material->uses = LAMBERT;
    specular_ball_material_phong->uses = LAMBERT | PHONG | TRANSPARENCY;
    specular_ball_material_blinn->uses = LAMBERT | BLINN;
    reflective_ball_material->uses = BLINN | REFLECTION;
    refractive_ball_material->uses = BLINN | REFRACTION;
    reflective_refractive_ball_material->uses = BLINN | REFLECTION | REFRACTION;

    specular_ball_material_phong->diffuse_color.z = 0.4f;
    diffuse_ball_material->diffuse_color.x = 0.3f;
    diffuse_ball_material->diffuse_color.z = 0.2f;
    diffuse_ball_material->diffuse_color.z = 0.7f;
}