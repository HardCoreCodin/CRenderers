#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"

#define saturate(value) max( 0.0f, min(value, 1.0f))

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 sdot(vec3* X, vec3* Y) { return saturate(dotVec3(X, Y));}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 sdotInv(vec3* X, vec3* Y) { return saturate(-dotVec3(X, Y));}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 schlickFresnel(f32 n1, f32 n2, f32 NdotL) {
    f32 R0 = (n1 - n2) / (n1 + n2);
    return R0 + (1 - R0)*powf(1 - NdotL, 5);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void reflect(vec3 *V, vec3 *N, f32 NdotV, vec3 *R) {
    vec3 NdotV2N;
    scaleVec3(N, -2 * NdotV, &NdotV2N);
    addVec3(V, &NdotV2N, R);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void refract(vec3* V, vec3* N, f32 NdotV, f32 n1_over_n2, vec3* out) {
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