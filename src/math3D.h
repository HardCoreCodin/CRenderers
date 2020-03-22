#pragma once
#pragma warning(disable : 4201)
#include "types.h"

typedef struct Vector3 {
    f32 x, y, z;
} Vector3;

typedef union {
    struct {
        f32 m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33;
    };
    struct {
        Vector3 i, j, k;
    };
} Matrix3x3;

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

Matrix3x3 yaw_matrix;
Matrix3x3 pitch_matrix;
Matrix3x3 roll_matrix;
Matrix3x3 rotation_matrix;

void setMatrix3x3ToIdentity(Matrix3x3* M) {
    M->m11 = M->m22 = M->m33 = 1.0f; 
    M->m12 = M->m21 = M->m23 = M->m32 = M->m13 = M->m31 = 0.0f;
}

inline void matmul(Matrix3x3* a, Matrix3x3* b, Matrix3x3* out) {
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

inline void imatmul(Matrix3x3* a, Matrix3x3* b) {
    f32 m11 = a->m11; f32 m21 = a->m21; f32 m31 = a->m31;
    f32 m12 = a->m12; f32 m22 = a->m22; f32 m32 = a->m32;
    f32 m13 = a->m13; f32 m23 = a->m23; f32 m33 = a->m33;

    a->m11 = m11*b->m11 + m12*b->m21 + m13*b->m31; // Row 1 | Column 1
    a->m12 = m11*b->m12 + m12*b->m22 + m13*b->m32; // Row 1 | Column 2
    a->m13 = m11*b->m13 + m12*b->m23 + m13*b->m33; // Row 1 | Column 3

    a->m21 = m21*b->m11 + m22*b->m21 + m23*b->m31; // Row 2 | Column 1
    a->m22 = m21*b->m12 + m22*b->m22 + m23*b->m32; // Row 2 | Column 2
    a->m23 = m21*b->m13 + m22*b->m23 + m23*b->m33; // Row 2 | Column 3

    a->m31 = m31*b->m11 + m32*b->m21 + m33*b->m31; // Row 3 | Column 1
    a->m32 = m31*b->m12 + m32*b->m22 + m33*b->m32; // Row 3 | Column 2
    a->m33 = m31*b->m13 + m32*b->m23 + m33*b->m33; // Row 3 | Column 3
}

inline void relative_yaw(f32 amount, Matrix3x3* matrix) {
    set_point_on_unit_circle(amount);

    f32 m11 = matrix->m11; f32 m21 = matrix->m21; f32 m31 = matrix->m31;
    f32 m13 = matrix->m13; f32 m23 = matrix->m23; f32 m33 = matrix->m33;

    matrix->m11 = c*m11 - s*m13; 
    matrix->m21 = c*m21 - s*m23; 
    matrix->m31 = c*m31 - s*m33; 

    matrix->m13 = c*m13 + s*m11;
    matrix->m23 = c*m23 + s*m21;
    matrix->m33 = c*m33 + s*m31;
};

inline void relative_pitch(f32 amount, Matrix3x3* matrix) {
    set_point_on_unit_circle(amount);

    f32 m12 = matrix->m12; f32 m22 = matrix->m22; f32 m32 = matrix->m32;
    f32 m13 = matrix->m13; f32 m23 = matrix->m23; f32 m33 = matrix->m33;

    matrix->m12 = c*m12 + s*m13; 
    matrix->m22 = c*m22 + s*m23; 
    matrix->m32 = c*m32 + s*m33;

    matrix->m13 = c*m13 - s*m12;
    matrix->m23 = c*m23 - s*m22;
    matrix->m33 = c*m33 - s*m32;
};

inline void relative_roll(f32 amount, Matrix3x3* matrix) {
    set_point_on_unit_circle(amount);

    f32 m11 = matrix->m11; f32 m21 = matrix->m21; f32 m31 = matrix->m31;
    f32 m12 = matrix->m12; f32 m22 = matrix->m22; f32 m32 = matrix->m32;
    
    matrix->m11 = c*m11 + s*m12; 
    matrix->m21 = c*m21 + s*m22; 
    matrix->m31 = c*m31 + s*m32; 

    matrix->m12 = c*m12 - s*m11;
    matrix->m22 = c*m22 - s*m21;
    matrix->m32 = c*m32 - s*m31;
};

inline void yaw(f32 amount) {
    set_point_on_unit_circle(amount);

    f32 m11 = yaw_matrix.m11; f32 m21 = yaw_matrix.m21; f32 m31 = yaw_matrix.m31;
    f32 m13 = yaw_matrix.m13; f32 m23 = yaw_matrix.m23; f32 m33 = yaw_matrix.m33;

    yaw_matrix.m11 = c*m11 - s*m13; 
    yaw_matrix.m21 = c*m21 - s*m23; 
    yaw_matrix.m31 = c*m31 - s*m33; 

    yaw_matrix.m13 = c*m13 + s*m11;
    yaw_matrix.m23 = c*m23 + s*m21;
    yaw_matrix.m33 = c*m33 + s*m31;
};

inline void pitch(f32 amount) {
    set_point_on_unit_circle(amount);

    f32 m12 = pitch_matrix.m12; f32 m22 = pitch_matrix.m22; f32 m32 = pitch_matrix.m32;
    f32 m13 = pitch_matrix.m13; f32 m23 = pitch_matrix.m23; f32 m33 = pitch_matrix.m33;

    pitch_matrix.m12 = c*m12 + s*m13; 
    pitch_matrix.m22 = c*m22 + s*m23; 
    pitch_matrix.m32 = c*m32 + s*m33;

    pitch_matrix.m13 = c*m13 - s*m12;
    pitch_matrix.m23 = c*m23 - s*m22;
    pitch_matrix.m33 = c*m33 - s*m32;
};

inline void roll(f32 amount) {
    set_point_on_unit_circle(amount);

    f32 m11 = roll_matrix.m11; f32 m21 = roll_matrix.m21; f32 m31 = roll_matrix.m31;
    f32 m12 = roll_matrix.m12; f32 m22 = roll_matrix.m22; f32 m32 = roll_matrix.m32;
    
    roll_matrix.m11 = c*m11 + s*m12; 
    roll_matrix.m21 = c*m21 + s*m22; 
    roll_matrix.m31 = c*m31 + s*m32; 

    roll_matrix.m12 = c*m12 - s*m11;
    roll_matrix.m22 = c*m22 - s*m21;
    roll_matrix.m32 = c*m32 - s*m31;
};

inline void set_yaw(f32 amount) {
    set_point_on_unit_circle(amount);

    yaw_matrix.m11 = yaw_matrix.m33 = c;
    yaw_matrix.m13 = +s;
    yaw_matrix.m31 = -s;
};

inline void set_pitch(f32 amount) {
    set_point_on_unit_circle(amount);

    pitch_matrix.m33 = pitch_matrix.m22 = c;
    pitch_matrix.m23 = -s;
    pitch_matrix.m32 = +s;
};

inline void set_roll(f32 amount) {
    set_point_on_unit_circle(amount);

    roll_matrix.m11 = roll_matrix.m22 = c;
    roll_matrix.m12 = -s;
    roll_matrix.m21 = +s;
};

void relative_rotate(Matrix3x3* matrix, f32 yaw_amount, f32 pitch_amount, f32 roll_amount) {
    if (yaw_amount) relative_yaw(yaw_amount, matrix);
    if (pitch_amount) relative_pitch(pitch_amount, matrix);
    if (roll_amount) relative_roll(roll_amount, matrix);
}

void absolute_rotate(f32 yaw_amount, f32 pitch_amount, f32 roll_amount) {
    if (roll_amount)
        set_roll(roll_amount);
    else
        setMatrix3x3ToIdentity(&roll_matrix);

    if (pitch_amount)
        set_pitch(pitch_amount);
    else
        setMatrix3x3ToIdentity(&pitch_matrix);

    if (yaw_amount)
        set_yaw(yaw_amount);
    else
        setMatrix3x3ToIdentity(&yaw_matrix);

    matmul(&roll_matrix, &pitch_matrix, &rotation_matrix);
    imatmul(&rotation_matrix, &yaw_matrix);
}

void rotate(f32 yaw_amount, f32 pitch_amount, f32 roll_amount) {
    if (yaw_amount) yaw(yaw_amount);
    if (pitch_amount) pitch(pitch_amount);
    if (roll_amount) {
        roll(roll_amount);
        matmul(&roll_matrix, &pitch_matrix, &rotation_matrix);
        imatmul(&rotation_matrix, &yaw_matrix);
    } else matmul(&pitch_matrix, &yaw_matrix, &rotation_matrix);
}

void init_math3D() {
    setMatrix3x3ToIdentity(&yaw_matrix);
    setMatrix3x3ToIdentity(&pitch_matrix);
    setMatrix3x3ToIdentity(&roll_matrix);
    setMatrix3x3ToIdentity(&rotation_matrix);
}