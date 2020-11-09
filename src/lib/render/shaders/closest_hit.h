#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/nodes/scene.h"
#include "intersection.h"

#define MAX_COLOR_VALUE 0xFF

vec3 white_color   = {1, 1, 1};
vec3 ambient_color = {10, 10, 20 };

#define IOR_AIR 1
#define IOR_GLASS 1.5f
f32 n1_over_n2_for_air_and_glass = IOR_AIR / IOR_GLASS;

inline void reflect(vec3 *V, vec3 *N, vec3 *R) {
    vec3 NdotV2N;
    f32 NdotV = max(-1.0f, min(dotVec3(N, V), 0.0f));
    scaleVec3(N, 2 * NdotV, &NdotV2N);
    subVec3(V, &NdotV2N, R);
}

inline void refract(vec3 *Rd, vec3 *N, vec3 *RRd, f32 n1_over_n2) {
    f32 a, b, y;
    vec3 X, Z, bX, yN;
    crossVec3(N, Rd, &Z);  // Z = N ⊕ Rd
    crossVec3(&Z, N, &X);  // X = Z ⊕ N
    a = dotVec3(Rd, &X);   // a = Rd • X
    b = n1_over_n2 * a;    // b = a•(n1/n2)
    y = sqrtf(1 - b*b); // y = √(1² - b²)
    scaleVec3(&X, b, &bX);  // bX = b•X
    scaleVec3(N, y, &yN);   // yN = y•N
    subVec3(&bX, &yN, RRd); // RRd = bX - yN
}

inline f32 schlickFresnel(f32 n1, f32 n2, f32 NdotL) {
    f32 R0 = (n1 - n2) / (n1 + n2);
    return R0 + (1 - R0)*powf(1 - NdotL, 5);
}

inline bool inShadow(vec3 *Rd, vec3* Ro, f32 light_distance) {
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    Sphere* sphere;
    f32 t, dt;
    vec3 _I, *I = &_I,
            _C, *C = &_C;

    for (sphere = scene.spheres; sphere != last_sphere; sphere++) {
        subVec3(sphere->position, Ro, C);
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

inline void shadeNormal(RayHit* hit, vec3* color) {
    vec3 normal_color;
    addVec3(&hit->normal, &white_color, &normal_color);
    scaleVec3(&normal_color, 4 * MAX_COLOR_VALUE / hit->distance, color);
}

inline void shadeDirection(vec3* ray_direction, vec3* color) {
    vec3 normal_color;
    addVec3(ray_direction, &white_color, &normal_color);
    scaleVec3(&normal_color, 0.5f * MAX_COLOR_VALUE, color);
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

    Light *last_light = scene.lights + scene.light_count;
    for (Light* light = scene.lights; light != last_light; light++) {
        subVec3(light->position, P, &L);
        light_distance = squaredLengthVec3(&L);
        light_attenuation = 1.0f / light_distance;
        light_distance = sqrtf(light_distance);
        iscaleVec3(&L, 1.0f / light_distance);
        if (inShadow(&L, P, light_distance)) continue;

        NdotL = max(0.0f, min(dotVec3(N, &L), 1.0f));
        light_intensity = light->intensity * light_attenuation * NdotL;

        scaleVec3(&light->color, light_intensity, &scaled_light_color);
        iaddVec3(color, &scaled_light_color);
    }
}

inline void shadePhong(RayHit* hit, vec3* color) {
    f32 light_intensity,
            light_attenuation,
            light_distance,
            NdotL,
            RdotL;
    vec3 L, R, scaled_light_color;
    vec3 *V = hit->ray_direction;
    vec3 *N = &hit->normal;
    vec3 *P = &hit->position;

    *color = ambient_color;

    reflect(V, N, &R);

    Light *last_light = scene.lights + scene.light_count;
    for (Light* light = scene.lights; light != last_light; light++) {
        subVec3(light->position, P, &L);
        light_distance = squaredLengthVec3(&L);
        light_attenuation = 1.0f / light_distance;
        light_distance = sqrtf(light_distance);
        iscaleVec3(&L, 1.0f / light_distance);
        if (inShadow(&L, P, light_distance)) continue;

        NdotL = max(0.0f, min(dotVec3(N, &L), 1.0f));
        RdotL = max(0.0f, min(dotVec3(&R, &L), 1.0f));

        light_intensity = light->intensity * light_attenuation * (NdotL + powf(RdotL, 4));

        scaleVec3(&light->color, light_intensity, &scaled_light_color);
        iaddVec3(color, &scaled_light_color);
    }
}

inline void shadeBlinn(RayHit* hit, vec3* color) {
    f32 light_intensity,
            light_attenuation,
            light_distance,
            NdotL,
            NdotH;
    vec3 L, H, scaled_light_color;
    vec3 *V = hit->ray_direction;
    vec3 *N = &hit->normal;
    vec3 *P = &hit->position;

    *color = ambient_color;

    Light *last_light = scene.lights + scene.light_count;
    for (Light* light = scene.lights; light != last_light; light++) {
        subVec3(light->position, P, &L);
        light_distance = squaredLengthVec3(&L);
        light_attenuation = 1.0f / light_distance;
        light_distance = sqrtf(light_distance);
        iscaleVec3(&L, 1.0f / light_distance);
        if (inShadow(&L, P, light_distance)) continue;

        NdotL = max(0.0f, min(dotVec3(N, &L), 1.0f));
        subVec3(&L, V, &H);
        norm3(&H);
        NdotH = max(0.0f, min(dotVec3(N, &H), 1.0f));

        light_intensity = light->intensity * light_attenuation * (NdotL + powf(NdotH, 16));

        scaleVec3(&light->color, light_intensity, &scaled_light_color);
        iaddVec3(color, &scaled_light_color);
    }
}

RayHit reflection_hit;

inline void shadeReflection(RayHit* hit, vec3* color) {
    f32 light_intensity,
            light_attenuation,
            light_distance,
            RdotL;
    vec3 L, R, scaled_light_color;
    vec3 *V = hit->ray_direction;
    vec3 *N = &hit->normal;
    vec3 *P = &hit->position;

    reflect(V, N, &R);

    reflection_hit.ray_direction = &R;
    reflection_hit.ray_origin = P;
    hitPlanes(&reflection_hit);
    hitSpheres(&reflection_hit, false);
    switch (reflection_hit.material_id) {
        case LAMBERT:       shadeLambert(&reflection_hit, color); break;
        case PHONG:           shadePhong(&reflection_hit, color); break;
        case BLINN:           shadeBlinn(&reflection_hit, color); break;
    }

    Light *last_light = scene.lights + scene.light_count;
    for (Light* light = scene.lights; light != last_light; light++) {
        subVec3(light->position, P, &L);
        light_distance = squaredLengthVec3(&L);
        light_attenuation = 1.0f / light_distance;
        light_distance = sqrtf(light_distance);
        iscaleVec3(&L, 1.0f / light_distance);
        if (inShadow(&L, P, light_distance)) continue;

        RdotL = max(0.0f, min(dotVec3(&R, &L), 1.0f));

        light_intensity = light->intensity * light_attenuation * powf(RdotL, 4);

        scaleVec3(&light->color, light_intensity, &scaled_light_color);
        iaddVec3(color, &scaled_light_color);
    }
}

RayHit refraction_hit;

inline void shadeRefraction(RayHit* hit, vec3* color) {
    f32 light_intensity,
            light_attenuation,
            light_distance,
            RdotL;
    vec3 L, R, scaled_light_color;
    vec3 *V = hit->ray_direction;
    vec3 *N = &hit->normal;
    vec3 *P = &hit->position;

    refract(V, N, &R, n1_over_n2_for_air_and_glass);

    refraction_hit.ray_direction = &R;
    refraction_hit.ray_origin = P;
    hitPlanes(&refraction_hit);
    hitSpheres(&refraction_hit, false);
    switch (refraction_hit.material_id) {
        case LAMBERT:       shadeLambert(&refraction_hit, color); break;
        case PHONG:           shadePhong(&refraction_hit, color); break;
        case BLINN:           shadeBlinn(&refraction_hit, color); break;
    }

    Light *last_light = scene.lights + scene.light_count;
    for (Light* light = scene.lights; light != last_light; light++) {
        subVec3(light->position, P, &L);
        light_distance = squaredLengthVec3(&L);
        light_attenuation = 1.0f / light_distance;
        light_distance = sqrtf(light_distance);
        iscaleVec3(&L, 1.0f / light_distance);
        if (inShadow(&L, P, light_distance)) continue;

        RdotL = max(0.0f, min(dotVec3(&R, &L), 1.0f));

        light_intensity = light->intensity * light_attenuation * powf(RdotL, 4);

        scaleVec3(&light->color, light_intensity, &scaled_light_color);
        iaddVec3(color, &scaled_light_color);
    }
}

inline void shadeRefractionDoubleSided(RayHit* hit, RayHit* farther_hit, vec3* color) {
    f32 light_intensity,
            light_attenuation,
            light_distance,
            RdotL,
            NdotV;
    vec3 L, RL1, RL2, RR1, RR2, scaled_light_color;
    vec3 *V = hit->ray_direction;
    vec3 *N1 = &hit->normal;
    vec3 *P1 = &hit->position;
    vec3 *N2 = &farther_hit->normal;
    vec3 *P2 = &farther_hit->position;

    f32 fresnel = 0;
    vec3 reflected_color;

    if (hit->distance) {
        reflect(V, N1, &RL1);
        refract(V, N1, &RR1, n1_over_n2_for_air_and_glass);

        reflect(&RR1, N2, &RL2);
        refract(&RR1, N2, &RR2, n1_over_n2_for_air_and_glass);

        reflected_color = ambient_color;
        Light *last_light = scene.lights + scene.light_count;
        for (Light* light = scene.lights; light != last_light; light++) {
            subVec3(light->position, P1, &L);
            light_distance = squaredLengthVec3(&L);
            light_attenuation = 1.0f / light_distance;
            light_distance = sqrtf(light_distance);
            iscaleVec3(&L, 1.0f / light_distance);
            if (inShadow(&L, P1, light_distance)) continue;

            RdotL = max(0.0f, min(dotVec3(&RL1, &L), 1.0f));

            light_intensity = light->intensity * light_attenuation * powf(RdotL, 4);

            scaleVec3(&light->color, light_intensity, &scaled_light_color);
            iaddVec3(&reflected_color, &scaled_light_color);
        }

        NdotV = max(0.0f, min(-dotVec3(N1, V), 1.0f));
        fresnel = max(0.0f, min(schlickFresnel(IOR_GLASS, IOR_AIR, NdotV), 1.0f));
        iscaleVec3(&reflected_color, fresnel);
    } else {
        reflect(V, N2, &RL2);
        refract(V, N2, &RR2, n1_over_n2_for_air_and_glass);
    }

    refraction_hit.ray_direction = &RR2;
    refraction_hit.ray_origin = P2;
    hitPlanes(&refraction_hit);
    hitSpheres(&refraction_hit, true);
    switch (refraction_hit.material_id) {
        case LAMBERT:       shadeLambert(&refraction_hit, color); break;
        case PHONG:           shadePhong(&refraction_hit, color); break;
        case BLINN:           shadeBlinn(&refraction_hit, color); break;
    }

    if (hit->distance) {
        iscaleVec3(color, 1 - fresnel);
        iaddVec3(color, &reflected_color);
    }
}

inline void shadeReflectionRefractionDoubleSided(RayHit* hit, RayHit* farther_hit, vec3* color) {
    f32 light_intensity,
            light_attenuation,
            light_distance,
            RdotL,
            NdotV;
    vec3 L, RL1, RL2, RR1, RR2, scaled_light_color;
    vec3 *V = hit->ray_direction;
    vec3 *N1 = &hit->normal;
    vec3 *P1 = &hit->position;
    vec3 *N2 = &farther_hit->normal;
    vec3 *P2 = &farther_hit->position;

    f32 fresnel = 0;
    vec3 reflected_color;

    if (hit->distance) {
        reflect(V, N1, &RL1);
        refract(V, N1, &RR1, n1_over_n2_for_air_and_glass);

        reflect(&RR1, N2, &RL2);
        refract(&RR1, N2, &RR2, n1_over_n2_for_air_and_glass);

        reflected_color = ambient_color;

        reflection_hit.ray_direction = &RL1;
        reflection_hit.ray_origin = P1;
        hitPlanes(&reflection_hit);
        hitSpheres(&reflection_hit, false);
        switch (reflection_hit.material_id) {
            case LAMBERT:       shadeLambert(&reflection_hit, &reflected_color); break;
            case PHONG:           shadePhong(&reflection_hit, &reflected_color); break;
            case BLINN:           shadeBlinn(&reflection_hit, &reflected_color); break;
        }

        Light *last_light = scene.lights + scene.light_count;
        for (Light* light = scene.lights; light != last_light; light++) {
            subVec3(light->position, P1, &L);
            light_distance = squaredLengthVec3(&L);
            light_attenuation = 1.0f / light_distance;
            light_distance = sqrtf(light_distance);
            iscaleVec3(&L, 1.0f / light_distance);
            if (inShadow(&L, P1, light_distance)) continue;

            RdotL = max(0.0f, min(dotVec3(&RL1, &L), 1.0f));

            light_intensity = light->intensity * light_attenuation * powf(RdotL, 4);

            scaleVec3(&light->color, light_intensity, &scaled_light_color);
            iaddVec3(color, &scaled_light_color);
        }

        NdotV = max(0.0f, min(-dotVec3(N1, V), 1.0f));
        fresnel = max(0.0f, min(schlickFresnel(IOR_GLASS, IOR_AIR, NdotV), 1.0f));
        iscaleVec3(&reflected_color, fresnel);
    } else {
        reflect(V, N2, &RL2);
        refract(V, N2, &RR2, n1_over_n2_for_air_and_glass);
    }

    refraction_hit.ray_direction = &RR2;
    refraction_hit.ray_origin = P2;
    hitPlanes(&refraction_hit);
    hitSpheres(&refraction_hit, true);
    switch (refraction_hit.material_id) {
        case LAMBERT:       shadeLambert(&refraction_hit, color); break;
        case PHONG:           shadePhong(&refraction_hit, color); break;
        case BLINN:           shadeBlinn(&refraction_hit, color); break;
    }

    if (hit->distance) {
        iscaleVec3(color, 1 - fresnel);
        iaddVec3(color, &reflected_color);
    }
}