#pragma once

#include "lib/core/types.h"
#include "lib/math/math1D.h"
#include "lib/math/math2D.h"
#include "lib/memory/allocators.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setMat3ToIdentity(mat3 *m) {
    m->X.x = 1; m->X.y = 0; m->X.z = 0;
    m->Y.x = 0; m->Y.y = 1; m->Y.z = 0;
    m->Z.x = 0; m->Z.y = 0; m->Z.z = 1;
}


#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void fillVec3(vec3* v, f32 value) {
    v->x = v->y = v->z = value;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void invertVec3(vec3* v) {
    v->x = -v->x;
    v->y = -v->y;
    v->z = -v->z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void approachVec3(vec3* current, vec3* target, f32 delta) {
    approach(&current->x, target->x, delta);
    approach(&current->y, target->y, delta);
    approach(&current->z, target->z, delta);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setPointOnUnitSphere(f32 s, f32 t, vec3* out) {
    f32 t_squared = t * t;
    f32 s_squared = s * s;
    f32 factor = 1 / ( t_squared + s_squared + 1);
    out->x = 2*s * factor;
    out->y = 2*t * factor;
    out->z = (t_squared + s_squared - 1) * t_squared;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool nonZeroVec3(vec3 *v) {
    return v->x != 0 ||
           v->y != 0 ||
           v->z != 0;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void subVec3(vec3* p1, vec3* p2, vec3* out) {
    out->x = p1->x - p2->x;
    out->y = p1->y - p2->y;
    out->z = p1->z - p2->z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void isubVec3(vec3* p1, vec3* p2) {
    p1->x -= p2->x;
    p1->y -= p2->y;
    p1->z -= p2->z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void addVec3(vec3* p1, vec3* p2, vec3* out) {
    out->x = p1->x + p2->x;
    out->y = p1->y + p2->y;
    out->z = p1->z + p2->z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void iaddVec3(vec3* p1, vec3* p2) {
    p1->x += p2->x;
    p1->y += p2->y;
    p1->z += p2->z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void mulVec3(vec3* p1, vec3* p2, vec3* out) {
    out->x = p1->x * p2->x;
    out->y = p1->y * p2->y;
    out->z = p1->z * p2->z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void imulVec3(vec3* p1, vec3* p2) {
    p1->x *= p2->x;
    p1->y *= p2->y;
    p1->z *= p2->z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void scaleVec3(vec3* p1, f32 factor, vec3* out) {
    out->x = p1->x * factor;
    out->y = p1->y * factor;
    out->z = p1->z * factor;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void iscaleVec3(vec3* v, f32 factor) {
    v->x *= factor;
    v->y *= factor;
    v->z *= factor;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void mulVec3Mat3(vec3* in, mat3* matrix, vec3* out) {
    out->x = in->x*matrix->X.x + in->y*matrix->Y.x + in->z*matrix->Z.x;
    out->y = in->x*matrix->X.y + in->y*matrix->Y.y + in->z*matrix->Z.y;
    out->z = in->x*matrix->X.z + in->y*matrix->Y.z + in->z*matrix->Z.z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void imulVec3Mat3(vec3* v, mat3* matrix) {
    vec3 X, Y, Z;
    scaleVec3(&matrix->X, v->x, &X);
    scaleVec3(&matrix->Y, v->y, &Y);
    scaleVec3(&matrix->Z, v->z, &Z);
    addVec3(&X, &Y, v);
    iaddVec3(v, &Z);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void crossVec3(vec3* p1, vec3* p2, vec3* out) {
    out->x = (p1->y * p2->z) - (p1->z * p2->y);
    out->y = (p1->z * p2->x) - (p1->x * p2->z);
    out->z = (p1->x * p2->y) - (p1->y * p2->x);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 dotVec3(vec3* p1, vec3* p2) {
    return (
            (p1->x * p2->x) +
            (p1->y * p2->y) +
            (p1->z * p2->z)
    );
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 squaredLengthVec3(vec3* v) {
    return (
            (v->x * v->x) +
            (v->y * v->y) +
            (v->z * v->z)
    );
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
f32 lengthVec3(vec3* v) {
    return sqrtf(squaredLengthVec3(v));
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void norm3(vec3* v) {
    iscaleVec3(v, 1.0f / lengthVec3(v));
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void transposeMat3(mat3* m, mat3* out) {
    out->X.x = m->X.x;  out->X.y = m->Y.x;  out->X.z = m->Z.x;
    out->Y.x = m->X.y;  out->Y.y = m->Y.y;  out->Y.z = m->Z.y;
    out->Z.x = m->X.z;  out->Z.y = m->Y.z;  out->Z.z = m->Z.z;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void mulMat3(mat3* a, mat3* b, mat3* out) {
    out->X.x = a->X.x*b->X.x + a->X.y*b->Y.x + a->X.z*b->Z.x; // Row 1 | Column 1
    out->X.y = a->X.x*b->X.y + a->X.y*b->Y.y + a->X.z*b->Z.y; // Row 1 | Column 2
    out->X.z = a->X.x*b->X.z + a->X.y*b->Y.z + a->X.z*b->Z.z; // Row 1 | Column 3

    out->Y.x = a->Y.x*b->X.x + a->Y.y*b->Y.x + a->Y.z*b->Z.x; // Row 2 | Column 1
    out->Y.y = a->Y.x*b->X.y + a->Y.y*b->Y.y + a->Y.z*b->Z.y; // Row 2 | Column 2
    out->Y.z = a->Y.x*b->X.z + a->Y.y*b->Y.z + a->Y.z*b->Z.z; // Row 2 | Column 3

    out->Z.x = a->Z.x*b->X.x + a->Z.y*b->Y.x + a->Z.z*b->Z.x; // Row 3 | Column 1
    out->Z.y = a->Z.x*b->X.y + a->Z.y*b->Y.y + a->Z.z*b->Z.y; // Row 3 | Column 2
    out->Z.z = a->Z.x*b->X.z + a->Z.y*b->Y.z + a->Z.z*b->Z.z; // Row 3 | Column 3
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool invMat3(mat3* m, mat3* out) {
    f32 m11 = m->X.x,  m12 = m->X.y,  m13 = m->X.z,
        m21 = m->Y.x,  m22 = m->Y.y,  m23 = m->Y.z,
        m31 = m->Z.x,  m32 = m->Z.y,  m33 = m->Z.z,

        c11 = m22*m33 -
              m23*m32,

        c12 = m13*m32 -
              m12*m33,

        c13 = m12*m23 -
              m13*m22,


        c21 = m23*m31 -
              m21*m33,

        c22 = m11*m33 -
              m13*m31,

        c23 = m13*m21 -
              m11*m23,


        c31 = m21*m32 -
              m22*m31,

        c32 = m12*m31 -
              m11*m32,

        c33 = m11*m22 -
              m12*m21,

        d = c11 + c12 + c13 +
            c21 + c22 + c23 +
            c31 + c32 + c33;

    if (!d) return false;

    d = 1 / d;

    out->X.x = d * c11;  out->X.y = d * c12;  out->X.z = d * c13;
    out->Y.x = d * c21;  out->Y.y = d * c22;  out->Y.z = d * c23;
    out->Z.x = d * c31;  out->Z.y = d * c32;  out->Z.z = d * c33;

    return true;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void imulMat3(mat3* a, mat3* b) {
    vec3 X = a->X;
    vec3 Y = a->Y;
    vec3 Z = a->Z;

    a->X.x = X.x * b->X.x + X.y * b->Y.x + X.z * b->Z.x; // Row 1 | Column 1
    a->X.y = X.x * b->X.y + X.y * b->Y.y + X.z * b->Z.y; // Row 1 | Column 2
    a->X.z = X.x * b->X.z + X.y * b->Y.z + X.z * b->Z.z; // Row 1 | Column 3

    a->Y.x = Y.x * b->X.x + Y.y * b->Y.x + Y.z * b->Z.x; // Row 2 | Column 1
    a->Y.y = Y.x * b->X.y + Y.y * b->Y.y + Y.z * b->Z.y; // Row 2 | Column 2
    a->Y.z = Y.x * b->X.z + Y.y * b->Y.z + Y.z * b->Z.z; // Row 2 | Column 3

    a->Z.x = Z.x * b->X.x + Z.y * b->Y.x + Z.z * b->Z.x; // Row 3 | Column 1
    a->Z.y = Z.x * b->X.y + Z.y * b->Y.y + Z.z * b->Z.y; // Row 3 | Column 2
    a->Z.z = Z.x * b->X.z + Z.y * b->Y.z + Z.z * b->Z.z; // Row 3 | Column 3
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void yawMat3(f32 amount, mat3* out) {
    vec2 xy = getPointOnUnitCircle(amount);

    vec3 X = out->X;
    vec3 Y = out->Y;
    vec3 Z = out->Z;

    out->X.x = xy.x * X.x - xy.y * X.z;
    out->Y.x = xy.x * Y.x - xy.y * Y.z;
    out->Z.x = xy.x * Z.x - xy.y * Z.z;

    out->X.z = xy.x * X.z + xy.y * X.x;
    out->Y.z = xy.x * Y.z + xy.y * Y.x;
    out->Z.z = xy.x * Z.z + xy.y * Z.x;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void pitchMat3(f32 amount, mat3* out) {
    vec2 xy = getPointOnUnitCircle(amount);

    vec3 X = out->X;
    vec3 Y = out->Y;
    vec3 Z = out->Z;

    out->X.y = xy.x * X.y + xy.y * X.z;
    out->Y.y = xy.x * Y.y + xy.y * Y.z;
    out->Z.y = xy.x * Z.y + xy.y * Z.z;

    out->X.z = xy.x * X.z - xy.y * X.y;
    out->Y.z = xy.x * Y.z - xy.y * Y.y;
    out->Z.z = xy.x * Z.z - xy.y * Z.y;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void rollMat3(f32 amount, mat3* out) {
    vec2 xy = getPointOnUnitCircle(amount);

    vec3 X = out->X;
    vec3 Y = out->Y;
    vec3 Z = out->Z;

    out->X.x = xy.x * X.x + xy.y * X.y;
    out->Y.x = xy.x * Y.x + xy.y * Y.y;
    out->Z.x = xy.x * Z.x + xy.y * Z.y;

    out->X.y = xy.x * X.y - xy.y * X.x;
    out->Y.y = xy.x * Y.y - xy.y * Y.x;
    out->Z.y = xy.x * Z.y - xy.y * Z.x;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setYawMat3(f32 yaw, mat3* yaw_matrix) {
    vec2 xy = getPointOnUnitCircle(yaw);

    yaw_matrix->X.x = yaw_matrix->Z.z = xy.x;
    yaw_matrix->X.z = +xy.y;
    yaw_matrix->Z.x = -xy.y;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setPitchMat3(f32 pitch, mat3* pitch_matrix) {
    vec2 xy = getPointOnUnitCircle(pitch);

    pitch_matrix->Z.z = pitch_matrix->Y.y = xy.x;
    pitch_matrix->Y.z = -xy.y;
    pitch_matrix->Z.y = +xy.y;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setRollMat3(f32 roll, mat3* roll_matrix) {
    vec2 xy = getPointOnUnitCircle(roll);

    roll_matrix->X.x = roll_matrix->Y.y = xy.x;
    roll_matrix->X.y = -xy.y;
    roll_matrix->Y.x = +xy.y;
}