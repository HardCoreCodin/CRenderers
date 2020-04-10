#pragma once

#include "lib/core/types.h"
#include "math2D.h"

Vector3 vec3;
Matrix3x3 mat3;

inline void fill3D(Vector3* vector, f32 value) {
    vector->x = vector->y = value, vector->z = value;
}

void setPointOnUnitSphere(f32 s, f32 t, Vector3* out) {
    f32 t_squared = t * t;
    f32 s_squared = s * s;
    f32 factor = 1 / ( t_squared + s_squared + 1);
    out->x = 2*s * factor;
    out->y = 2*t * factor;
    out->z = (t_squared + s_squared - 1) * t_squared;
}

inline void sub3D(Vector3* p1, Vector3* p2, Vector3* out) {
    out->x = p1->x - p2->x;
    out->y = p1->y - p2->y;
    out->z = p1->z - p2->z;
}
inline void isub3D(Vector3* p1, Vector3* p2) {
    p1->x -= p2->x;
    p1->y -= p2->y;
    p1->z -= p2->z;
}
inline void add3D(Vector3* p1, Vector3* p2, Vector3* out) {
    out->x = p1->x + p2->x;
    out->y = p1->y + p2->y;
    out->z = p1->z + p2->z;
}
inline void iadd3D(Vector3* p1, Vector3* p2) {
    p1->x += p2->x;
    p1->y += p2->y;
    p1->z += p2->z;
}
inline void scale3D(Vector3* p1, f32 factor, Vector3* out) {
    out->x = p1->x * factor;
    out->y = p1->y * factor;
    out->z = p1->z * factor;
}
inline void iscale3D(Vector3* v, f32 factor) {
    v->x *= factor;
    v->y *= factor;
    v->z *= factor;
}
inline void mul3D(Vector3* in, Matrix3x3* matrix, Vector3* out) {
    out->x = in->x*matrix->m11 + in->y*matrix->m21 + in->z*matrix->m31;
    out->y = in->x*matrix->m12 + in->y*matrix->m22 + in->z*matrix->m32;
    out->z = in->x*matrix->m13 + in->y*matrix->m23 + in->z*matrix->m33;    
}
inline void imul3D(Vector3* v, Matrix3x3* matrix) {
    f32 x = v->x;
    f32 y = v->y;
    f32 z = v->z;

    v->x = x*matrix->m11 + y*matrix->m21 + z*matrix->m31;
    v->y = x*matrix->m12 + y*matrix->m22 + z*matrix->m32;
    v->z = x*matrix->m13 + y*matrix->m23 + z*matrix->m33;    
}
inline void cross3D(Vector3* p1, Vector3* p2, Vector3* out) {
    out->x = (p1->y * p2->z) - (p1->z * p2->y);
    out->y = (p1->z * p2->x) - (p1->x * p2->z);
    out->z = (p1->x * p2->y) - (p1->y * p2->x);
}
inline f32 dot3D(Vector3* p1, Vector3* p2) {
    return (
        (p1->x * p2->x) +
        (p1->y * p2->y) +
        (p1->z * p2->z)
    );
}
inline f32 squaredLength3D(Vector3* v) {
    return (
        (v->x * v->x) +
        (v->y * v->y) +
        (v->z * v->z)
    );
}

void setMatrix3x3ToIdentity(Matrix3x3* M) {
    M->m11 = M->m22 = M->m33 = 1.0f; 
    M->m12 = M->m21 = M->m23 = M->m32 = M->m13 = M->m31 = 0.0f;
}

void transposeMatrix3D(Matrix3x3* M, Matrix3x3* O) {
    O->m11 = M->m11; O->m22 = M->m22; O->m33 = M->m33;
    O->m12 = M->m21; O->m21 = M->m12;
    O->m13 = M->m31; O->m31 = M->m13;
    O->m23 = M->m32; O->m32 = M->m23;
}

inline void matMul3D(Matrix3x3* a, Matrix3x3* b, Matrix3x3* out) {
    out->m11 = a->m11*b->m11 + a->m12*b->m21 + a->m13*b->m31; // Row 1 | Column 1
    out->m12 = a->m11*b->m12 + a->m12*b->m22 + a->m13*b->m32; // Row 1 | Column 2
    out->m13 = a->m11*b->m13 + a->m12*b->m23 + a->m13*b->m33; // Row 1 | Column 3

    out->m21 = a->m21*b->m11 + a->m22*b->m21 + a->m23*b->m31; // Row 2 | Column 1
    out->m22 = a->m21*b->m12 + a->m22*b->m22 + a->m23*b->m32; // Row 2 | Column 2
    out->m23 = a->m21*b->m13 + a->m22*b->m23 + a->m23*b->m33; // Row 2 | Column 3

    out->m31 = a->m31*b->m11 + a->m32*b->m21 + a->m33*b->m31; // Row 3 | Column 1
    out->m32 = a->m31*b->m12 + a->m32*b->m22 + a->m33*b->m32; // Row 3 | Column 2
    out->m33 = a->m31*b->m13 + a->m32*b->m23 + a->m33*b->m33; // Row 3 | Column 3
}

inline void imatMul3D(Matrix3x3* a, Matrix3x3* b) {
    mat3.m11 = a->m11; mat3.m21 = a->m21; mat3.m31 = a->m31;
    mat3.m12 = a->m12; mat3.m22 = a->m22; mat3.m32 = a->m32;
    mat3.m13 = a->m13; mat3.m23 = a->m23; mat3.m33 = a->m33;

    a->m11 = mat3.m11*b->m11 + mat3.m12*b->m21 + mat3.m13*b->m31; // Row 1 | Column 1
    a->m12 = mat3.m11*b->m12 + mat3.m12*b->m22 + mat3.m13*b->m32; // Row 1 | Column 2
    a->m13 = mat3.m11*b->m13 + mat3.m12*b->m23 + mat3.m13*b->m33; // Row 1 | Column 3

    a->m21 = mat3.m21*b->m11 + mat3.m22*b->m21 + mat3.m23*b->m31; // Row 2 | Column 1
    a->m22 = mat3.m21*b->m12 + mat3.m22*b->m22 + mat3.m23*b->m32; // Row 2 | Column 2
    a->m23 = mat3.m21*b->m13 + mat3.m22*b->m23 + mat3.m23*b->m33; // Row 2 | Column 3

    a->m31 = mat3.m31*b->m11 + mat3.m32*b->m21 + mat3.m33*b->m31; // Row 3 | Column 1
    a->m32 = mat3.m31*b->m12 + mat3.m32*b->m22 + mat3.m33*b->m32; // Row 3 | Column 2
    a->m33 = mat3.m31*b->m13 + mat3.m32*b->m23 + mat3.m33*b->m33; // Row 3 | Column 3
}

inline void relativeYaw3D(f32 yaw, Matrix3x3* yaw_matrix) {
    setPointOnUnitCircle(yaw);

    mat3.m11 = yaw_matrix->m11; mat3.m21 = yaw_matrix->m21; mat3.m31 = yaw_matrix->m31;
    mat3.m13 = yaw_matrix->m13; mat3.m23 = yaw_matrix->m23; mat3.m33 = yaw_matrix->m33;

    yaw_matrix->m11 = vec2.x * mat3.m11 - vec2.y * mat3.m13;
    yaw_matrix->m21 = vec2.x * mat3.m21 - vec2.y * mat3.m23;
    yaw_matrix->m31 = vec2.x * mat3.m31 - vec2.y * mat3.m33;

    yaw_matrix->m13 = vec2.x * mat3.m13 + vec2.y * mat3.m11;
    yaw_matrix->m23 = vec2.x * mat3.m23 + vec2.y * mat3.m21;
    yaw_matrix->m33 = vec2.x * mat3.m33 + vec2.y * mat3.m31;
};

inline void relativePitch3D(f32 pitch, Matrix3x3* pitch_matrix) {
    setPointOnUnitCircle(pitch);

    mat3.m12 = pitch_matrix->m12; mat3.m22 = pitch_matrix->m22; mat3.m32 = pitch_matrix->m32;
    mat3.m13 = pitch_matrix->m13; mat3.m23 = pitch_matrix->m23; mat3.m33 = pitch_matrix->m33;

    pitch_matrix->m12 = vec2.x * mat3.m12 + vec2.y * mat3.m13;
    pitch_matrix->m22 = vec2.x * mat3.m22 + vec2.y * mat3.m23;
    pitch_matrix->m32 = vec2.x * mat3.m32 + vec2.y * mat3.m33;

    pitch_matrix->m13 = vec2.x * mat3.m13 - vec2.y * mat3.m12;
    pitch_matrix->m23 = vec2.x * mat3.m23 - vec2.y * mat3.m22;
    pitch_matrix->m33 = vec2.x * mat3.m33 - vec2.y * mat3.m32;
};

inline void relativeRoll3D(f32 roll, Matrix3x3* roll_matrix) {
    setPointOnUnitCircle(roll);

    mat3.m11 = roll_matrix->m11; mat3.m21 = roll_matrix->m21; mat3.m31 = roll_matrix->m31;
    mat3.m12 = roll_matrix->m12; mat3.m22 = roll_matrix->m22; mat3.m32 = roll_matrix->m32;

    roll_matrix->m11 = vec2.x * mat3.m11 + vec2.y * mat3.m12;
    roll_matrix->m21 = vec2.x * mat3.m21 + vec2.y * mat3.m22;
    roll_matrix->m31 = vec2.x * mat3.m31 + vec2.y * mat3.m32;

    roll_matrix->m12 = vec2.x * mat3.m12 - vec2.y * mat3.m11;
    roll_matrix->m22 = vec2.x * mat3.m22 - vec2.y * mat3.m21;
    roll_matrix->m32 = vec2.x * mat3.m32 - vec2.y * mat3.m31;
};

inline void yaw3D(f32 yaw, Matrix3x3* yaw_matrix) {
    setPointOnUnitCircle(yaw);

    mat3.m11 = yaw_matrix->m11; mat3.m21 = yaw_matrix->m21; mat3.m31 = yaw_matrix->m31;
    mat3.m13 = yaw_matrix->m13; mat3.m23 = yaw_matrix->m23; mat3.m33 = yaw_matrix->m33;

    yaw_matrix->m11 = vec2.x * mat3.m11 - vec2.y * mat3.m13;
    yaw_matrix->m21 = vec2.x * mat3.m21 - vec2.y * mat3.m23;
    yaw_matrix->m31 = vec2.x * mat3.m31 - vec2.y * mat3.m33;

    yaw_matrix->m13 = vec2.x * mat3.m13 + vec2.y * mat3.m11;
    yaw_matrix->m23 = vec2.x * mat3.m23 + vec2.y * mat3.m21;
    yaw_matrix->m33 = vec2.x * mat3.m33 + vec2.y * mat3.m31;
};

inline void pitch3D(f32 pitch, Matrix3x3* pitch_matrix) {
    setPointOnUnitCircle(pitch);

    mat3.m12 = pitch_matrix->m12; mat3.m22 = pitch_matrix->m22; mat3.m32 = pitch_matrix->m32;
    mat3.m13 = pitch_matrix->m13; mat3.m23 = pitch_matrix->m23; mat3.m33 = pitch_matrix->m33;

    pitch_matrix->m12 = vec2.x * mat3.m12 + vec2.y * mat3.m13;
    pitch_matrix->m22 = vec2.x * mat3.m22 + vec2.y * mat3.m23;
    pitch_matrix->m32 = vec2.x * mat3.m32 + vec2.y * mat3.m33;

    pitch_matrix->m13 = vec2.x * mat3.m13 - vec2.y * mat3.m12;
    pitch_matrix->m23 = vec2.x * mat3.m23 - vec2.y * mat3.m22;
    pitch_matrix->m33 = vec2.x * mat3.m33 - vec2.y * mat3.m32;
};

inline void roll3D(f32 roll, Matrix3x3* roll_matrix) {
    setPointOnUnitCircle(roll);

    mat3.m11 = roll_matrix->m11; mat3.m21 = roll_matrix->m21; mat3.m31 = roll_matrix->m31;
    mat3.m12 = roll_matrix->m12; mat3.m22 = roll_matrix->m22; mat3.m32 = roll_matrix->m32;
    
    roll_matrix->m11 = vec2.x * mat3.m11 + vec2.y * mat3.m12;
    roll_matrix->m21 = vec2.x * mat3.m21 + vec2.y * mat3.m22;
    roll_matrix->m31 = vec2.x * mat3.m31 + vec2.y * mat3.m32;

    roll_matrix->m12 = vec2.x * mat3.m12 - vec2.y * mat3.m11;
    roll_matrix->m22 = vec2.x * mat3.m22 - vec2.y * mat3.m21;
    roll_matrix->m32 = vec2.x * mat3.m32 - vec2.y * mat3.m31;
};

inline void setYaw3D(f32 yaw, Matrix3x3* yaw_matrix) {
    setPointOnUnitCircle(yaw);

    yaw_matrix->m11 = yaw_matrix->m33 = vec2.x;
    yaw_matrix->m13 = +vec2.y;
    yaw_matrix->m31 = -vec2.y;
};

inline void setPitch3D(f32 pitch, Matrix3x3* pitch_matrix) {
    setPointOnUnitCircle(pitch);

    pitch_matrix->m33 = pitch_matrix->m22 = vec2.x;
    pitch_matrix->m23 = -vec2.y;
    pitch_matrix->m32 = +vec2.y;
};

inline void setRoll3D(f32 roll, Matrix3x3* roll_matrix) {
    setPointOnUnitCircle(roll);

    roll_matrix->m11 = roll_matrix->m22 = vec2.x;
    roll_matrix->m12 = -vec2.y;
    roll_matrix->m21 = +vec2.y;
};

void rotateRelative3D(f32 yaw, f32 pitch, f32 roll, Matrix3x3* rotation_matrix) {
    if (yaw) relativeYaw3D(yaw, rotation_matrix);
    if (pitch) relativePitch3D(pitch, rotation_matrix);
    if (roll) relativeRoll3D(roll, rotation_matrix);
}

void rotateAbsolute3D(
        f32 yaw,
        f32 pitch,
        f32 roll,

        Matrix3x3* yaw_matrix,
        Matrix3x3* pitch_matrix,
        Matrix3x3* roll_matrix,

        Matrix3x3* rotation_matrix
) {
    if (roll)
        setRoll3D(roll, roll_matrix);
    else
        setMatrix3x3ToIdentity(roll_matrix);

    if (pitch)
        setPitch3D(pitch, pitch_matrix);
    else
        setMatrix3x3ToIdentity(pitch_matrix);

    if (yaw)
        setYaw3D(yaw, yaw_matrix);
    else
        setMatrix3x3ToIdentity(yaw_matrix);

    matMul3D(roll_matrix, pitch_matrix, rotation_matrix);
    imatMul3D(rotation_matrix, yaw_matrix);
}

void rotate3D(f32 yaw, f32 pitch, f32 roll, Transform3D* transform) {
    if (yaw) yaw3D(yaw, transform->yaw);
    if (pitch) pitch3D(pitch, transform->pitch);
    if (roll) {
        roll3D(roll, transform->roll);
        matMul3D(transform->roll, transform->pitch, transform->rotation);
        imatMul3D(transform->rotation, transform->yaw);
    } else
        matMul3D(transform->pitch, transform->yaw, transform->rotation);
}
