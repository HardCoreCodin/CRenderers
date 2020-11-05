#pragma once

#include "lib/core/types.h"
#include "lib/nodes/scene.h"
#include "lib/math/math3D.h"

inline void shadeBlinn(RayHit* closestHit, Pixel* pixel) {
    f32 light_intensity,
        light_attenuation,
        light_distance,
        NdotL,
        NdotH;
    vec3 L, H;
    vec3 *V = closestHit->ray_direction;
    vec3 *N = &closestHit->normal;
    vec3 *P = &closestHit->position;
    f32 r = 10;
    f32 g = 10;
    f32 b = 20;

    bool in_shadow;
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    Sphere* sphere;
    f32 r2, // The radius of the current sphere (and it's square)
            d, d2, // The distance from the origin to the position of the current intersection (and it's square)
            o2c = 0, // The distance from the ray's origin to a position along the ray closest to the current sphere's center
            r2_minus_d2 = 0; // The square of the distance from that position to the current intersection position
    vec3 _p, _t, _s;
    vec3 *s = &_s, // The center position of the sphere of the current intersection
            *p = &_p, // The position of the current intersection of the ray with the spheres
            *t = &_t;

    Light *last_light = scene.lights + scene.light_count;
    for (Light* light = scene.lights; light != last_light; light++) {
        subVec3(light->view_position, P, &L);
        light_distance = squaredLengthVec3(&L);
        light_attenuation = 1.0f / light_distance;
        light_distance = sqrtf(light_distance);
        iscaleVec3(&L, 1.0f / light_distance);

        in_shadow = false;
        for (sphere = scene.spheres; sphere != last_sphere; sphere++) {
            subVec3(sphere->view_position, P, s);
            r2 = sphere->radius * sphere->radius;
            o2c = dotVec3(&L, s);
            if (o2c > 0 && o2c < light_distance) {
                scaleVec3(&L, o2c, p);
                subVec3(s, p, t);
                d2 = squaredLengthVec3(t);
                if (d2 <= r2) {
                    r2_minus_d2 = r2 - d2;
                    d = o2c - r2_minus_d2;
                    if (d > 0) {
                        in_shadow = true;
                        break;
                    }
                }
            }
        }

        if (in_shadow) continue;

        NdotL = max(0.0f, min(dotVec3(N, &L), 1.0f));

        subVec3(&L, V, &H);
        norm3(&H);
        NdotH = max(0.0f, min(dotVec3(N, &H), 1.0f));

        light_intensity = light->intensity * light_attenuation * (NdotL + powf(NdotH, 16));

        r += light_intensity * (f32)light->color.R;
        g += light_intensity * (f32)light->color.G;
        b += light_intensity * (f32)light->color.B;
    }

    pixel->color.R = r > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)r;
    pixel->color.G = g > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)g;
    pixel->color.B = b > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)b;
}