#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
vec2 getUV(vec3 *direction) {
    vec2 uv;

    f32 x = direction->x;
    f32 y = direction->y;
    f32 z = direction->z;

    f32 z_over_x = x ? (z / x) : 2;
    f32 y_over_x = x ? (y / x) : 2;
    if (z_over_x <=  1 &&
        z_over_x >= -1 &&
        y_over_x <=  1 &&
        y_over_x >= -1) { // Right or Left
        uv.x = z_over_x;
        uv.y = x > 0 ? y_over_x : -y_over_x;
    } else {
        f32 x_over_z = z ? (x / z) : 2;
        f32 y_over_z = z ? (y / z) : 2;
        if (x_over_z <=  1 &&
            x_over_z >= -1 &&
            y_over_z <=  1 &&
            y_over_z >= -1) { // Front or Back:
            uv.x = -x_over_z;
            uv.y = z > 0 ? y_over_z : -y_over_z;
        } else {
            uv.x = x / (y > 0 ? y : -y);
            uv.y = z / y;
        }
    }

    uv.x += 1;  uv.x /= 2;
    uv.y += 1;  uv.y /= 2;

    return uv;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setRayHitPosition(vec3 *origin, vec3 *direction, f32 distance, vec3 *position) {
    scaleVec3(direction, distance, position);
    iaddVec3(position, origin);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setRayHitDirection(vec3 *position, vec3 *center, vec3 *direction) {
    subVec3(position, center, direction);
    norm3(direction);
}

// Spherical UV:

// Ru / 1 = z / x  :  x > 0, -1 <= Ru <= 1
// Rv / 1 = y / x  :  x > 0, -1 <= Rv <= 1

// Lu / 1 = -z / -x  :  x < 0, -1 <= Lu <= 1
// Lv / 1 = y / -x   :  x < 0, -1 <= Lv <= 1


// Fu / 1 = x / -z  :  z < 0, -1 <= Ru <= 1
// Fv / 1 = y / -z  :  z < 0, -1 <= Rv <= 1

// Ku / 1 = -x / z  :  z > 0, -1 <= Lu <= 1
// Kv / 1 = y / z   :  z > 0, -1 <= Lv <= 1


// Tu / 1 = x / y  :  y > 0, -1 <= Tu <= 1
// Tv / 1 = z / y  :  y > 0, -1 <= Tv <= 1

// Bu / 1 = x / -y  : y < 0,  -1 <= Bu <= 1
// Bv / 1 = -z / -y  :  y < 0, -1 <= Bv <= 1