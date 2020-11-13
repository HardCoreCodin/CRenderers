#pragma once

#include "lib/core/types.h"
#include "lib/math/math1D.h"
#include "lib/math/math2D.h"
#include "lib/memory/allocators.h"

inline void setMat3ToIdentity(mat3 *m) {
    m->X.x = 1; m->X.y = 0; m->X.z = 0;
    m->Y.x = 0; m->Y.y = 1; m->Y.z = 0;
    m->Z.x = 0; m->Z.y = 0; m->Z.z = 1;
}

inline void fillVec3(vec3* v, f32 value) {
    v->x = v->y = v->z = value;
}

inline void approachVec3(vec3* current, vec3* target, f32 delta) {
    approach(&current->x, target->x, delta);
    approach(&current->y, target->y, delta);
    approach(&current->z, target->z, delta);
}

void setPointOnUnitSphere(f32 s, f32 t, vec3* out) {
    f32 t_squared = t * t;
    f32 s_squared = s * s;
    f32 factor = 1 / ( t_squared + s_squared + 1);
    out->x = 2*s * factor;
    out->y = 2*t * factor;
    out->z = (t_squared + s_squared - 1) * t_squared;
}

inline bool nonZeroVec3(vec3 *v) {
    return v->x != 0 ||
           v->y != 0 ||
           v->z !=0;
}

inline void subVec3(vec3* p1, vec3* p2, vec3* out) {
    out->x = p1->x - p2->x;
    out->y = p1->y - p2->y;
    out->z = p1->z - p2->z;
}
inline void isubVec3(vec3* p1, vec3* p2) {
    p1->x -= p2->x;
    p1->y -= p2->y;
    p1->z -= p2->z;
}
inline void addVec3(vec3* p1, vec3* p2, vec3* out) {
    out->x = p1->x + p2->x;
    out->y = p1->y + p2->y;
    out->z = p1->z + p2->z;
}
inline void iaddVec3(vec3* p1, vec3* p2) {
    p1->x += p2->x;
    p1->y += p2->y;
    p1->z += p2->z;
}
inline void scaleVec3(vec3* p1, f32 factor, vec3* out) {
    out->x = p1->x * factor;
    out->y = p1->y * factor;
    out->z = p1->z * factor;
}
inline void iscaleVec3(vec3* v, f32 factor) {
    v->x *= factor;
    v->y *= factor;
    v->z *= factor;
}
inline void mulVec3Mat3(vec3* in, mat3* matrix, vec3* out) {
    out->x = in->x*matrix->X.x + in->y*matrix->Y.x + in->z*matrix->Z.x;
    out->y = in->x*matrix->X.y + in->y*matrix->Y.y + in->z*matrix->Z.y;
    out->z = in->x*matrix->X.z + in->y*matrix->Y.z + in->z*matrix->Z.z;
}
inline void imulVec3Mat3(vec3* v, mat3* matrix) {
    f32 x = v->x;
    f32 y = v->y;
    f32 z = v->z;

    v->x = x*matrix->X.x + y*matrix->Y.x + z*matrix->Z.x;
    v->y = x*matrix->X.y + y*matrix->Y.y + z*matrix->Z.y;
    v->z = x*matrix->X.z + y*matrix->Y.z + z*matrix->Z.z;
}
inline void crossVec3(vec3* p1, vec3* p2, vec3* out) {
    out->x = (p1->y * p2->z) - (p1->z * p2->y);
    out->y = (p1->z * p2->x) - (p1->x * p2->z);
    out->z = (p1->x * p2->y) - (p1->y * p2->x);
}
inline f32 dotVec3(vec3* p1, vec3* p2) {
    return (
            (p1->x * p2->x) +
            (p1->y * p2->y) +
            (p1->z * p2->z)
    );
}
inline f32 squaredLengthVec3(vec3* v) {
    return (
            (v->x * v->x) +
            (v->y * v->y) +
            (v->z * v->z)
    );
}

inline void norm3(vec3* v) {
    iscaleVec3(v, 1.0f / sqrtf(squaredLengthVec3(v)));
}

inline void transposeMat3(mat3* m, mat3* out) {
    out->X.x = m->X.x;  out->X.y = m->Y.x;  out->X.z = m->Z.x;
    out->Y.x = m->X.y;  out->Y.y = m->Y.y;  out->Y.z = m->Z.y;
    out->Z.x = m->X.z;  out->Z.y = m->Y.z;  out->Z.z = m->Z.z;
}

inline void mulMat3(mat3* a, mat3* b, mat3* out) {
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

inline void imulMat3(mat3* a, mat3* b) {
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

inline void yawMat3(f32 amount, mat3* out) {
    f32 s, c;
    getPointOnUnitCircle(amount, &s, &c);

    vec3 X = out->X;
    vec3 Y = out->Y;
    vec3 Z = out->Z;

    out->X.x = c * X.x - s * X.z;
    out->Y.x = c * Y.x - s * Y.z;
    out->Z.x = c * Z.x - s * Z.z;

    out->X.z = c * X.z + s * X.x;
    out->Y.z = c * Y.z + s * Y.x;
    out->Z.z = c * Z.z + s * Z.x;
}

inline void pitchMat3(f32 amount, mat3* out) {
    f32 s, c;
    getPointOnUnitCircle(amount, &s, &c);

    vec3 X = out->X;
    vec3 Y = out->Y;
    vec3 Z = out->Z;

    out->X.y = c * X.y + s * X.z;
    out->Y.y = c * Y.y + s * Y.z;
    out->Z.y = c * Z.y + s * Z.z;

    out->X.z = c * X.z - s * X.y;
    out->Y.z = c * Y.z - s * Y.y;
    out->Z.z = c * Z.z - s * Z.y;
}

inline void rollMat3(f32 amount, mat3* out) {
    f32 s, c;
    getPointOnUnitCircle(amount, &s, &c);

    vec3 X = out->X;
    vec3 Y = out->Y;
    vec3 Z = out->Z;

    out->X.x = c * X.x + s * X.y;
    out->Y.x = c * Y.x + s * Y.y;
    out->Z.x = c * Z.x + s * Z.y;

    out->X.y = c * X.y - s * X.x;
    out->Y.y = c * Y.y - s * Y.x;
    out->Z.y = c * Z.y - s * Z.x;
}

inline void setYawMat3(f32 yaw, mat3* yaw_matrix) {
    f32 s, c;
    getPointOnUnitCircle(yaw, &s, &c);

    yaw_matrix->X.x = yaw_matrix->Z.z = c;
    yaw_matrix->X.z = +s;
    yaw_matrix->Z.x = -s;
};

inline void setPitchMat3(f32 pitch, mat3* pitch_matrix) {
    f32 s, c;
    getPointOnUnitCircle(pitch, &s, &c);

    pitch_matrix->Z.z = pitch_matrix->Y.y = c;
    pitch_matrix->Y.z = -s;
    pitch_matrix->Z.y = +s;
};

inline void setRollMat3(f32 roll, mat3* roll_matrix) {
    f32 s, c;
    getPointOnUnitCircle(roll, &s, &c);

    roll_matrix->X.x = roll_matrix->Y.y = c;
    roll_matrix->X.y = -s;
    roll_matrix->Y.x = +s;
};
