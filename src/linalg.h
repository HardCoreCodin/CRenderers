#pragma once
#pragma warning(disable : 4201)
#include "types.h"

f32 m11, m12, m13, 
    m21, m22, m23, 
    m31, m32, m33;

typedef struct Vector2 {
	f32 x, y;
} Vector2;

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

Matrix3x3 yaw_matrix;
Matrix3x3 pitch_matrix;
Matrix3x3 roll_matrix;
Matrix3x3 rotation_matrix;

void setMatrixToIdentity(Matrix3x3* M) {
    M->m11 = M->m22 = M->m33 = 1.0f; 
    M->m12 = M->m21 = M->m23 = M->m32 = M->m13 = M->m31 = 0.0f;
}

inline void sub(Vector3* p1, Vector3* p2, Vector3* out) {
    out->x = p1->x - p2->x;
    out->y = p1->y - p2->y;
    out->z = p1->z - p2->z;
}
inline void isub(Vector3* p1, Vector3* p2) {
    p1->x -= p2->x;
    p1->y -= p2->y;
    p1->z -= p2->z;
}
inline void add(Vector3* p1, Vector3* p2, Vector3* out) {
    out->x = p1->x + p2->x;
    out->y = p1->y + p2->y;
    out->z = p1->z + p2->z;
}
inline void iadd(Vector3* p1, Vector3* p2) {
    p1->x += p2->x;
    p1->y += p2->y;
    p1->z += p2->z;
}
inline void scale(Vector3* p1, f32 factor, Vector3* out) {
    out->x = p1->x * factor;
    out->y = p1->y * factor;
    out->z = p1->z * factor;
}
inline void iscale(Vector3* v, f32 factor) {
    v->x *= factor;
    v->y *= factor;
    v->z *= factor;
}
inline void idiv(Vector3* v, f32 factor) {
    v->x /= factor;
    v->y /= factor;
    v->z /= factor;
}
inline void cross(Vector3* p1, Vector3* p2, Vector3* out) {
    out->x = (p1->y * p2->z) - (p1->z * p2->y);
    out->y = (p1->z * p2->x) - (p1->x * p2->z);
    out->z = (p1->x * p2->y) - (p1->y * p2->x);
}
inline f32 dot(Vector3* p1, Vector3* p2) {
    return (
            (p1->x * p2->x) +
            (p1->y * p2->y) +
            (p1->z * p2->z)
    );
}
inline f32 length_squared(Vector3* v) {
    return (
            (v->x * v->x) +
            (v->y * v->y) +
            (v->z * v->z)
    );
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
    m11 = a->m11; m21 = a->m21; m31 = a->m31;
    m12 = a->m12; m22 = a->m22; m32 = a->m32;
    m13 = a->m13; m23 = a->m23; m33 = a->m33;

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

typedef struct Trig {
    f32 sin, cos;
} Trig;

static Trig trig;

void set_sin_cos(f32 t) {
    f32 t2 = t * t;
    
    trig.sin = 2 * t;
    trig.cos = 1 - t2;
    
    t2 = 1 / (t2 + 1);

    trig.sin *= t2;
    trig.cos *= t2;
}

inline void relative_yaw(f32 amount, Matrix3x3* matrix) {
    set_sin_cos(amount);

    m11 = matrix->m11; m21 = matrix->m21; m31 = matrix->m31;
    m12 = matrix->m12; m22 = matrix->m22; m32 = matrix->m32;
    m13 = matrix->m13; m23 = matrix->m23; m33 = matrix->m33;

    matrix->m11 = m11 * trig.cos - m13 * trig.sin; 
    matrix->m21 = m21 * trig.cos - m23 * trig.sin; 
    matrix->m31 = m31 * trig.cos - m33 * trig.sin; 

    matrix->m13 = m13 * trig.cos + m11 * trig.sin;
    matrix->m23 = m23 * trig.cos + m21 * trig.sin;
    matrix->m33 = m33 * trig.cos + m31 * trig.sin;
};

inline void relative_pitch(f32 amount, Matrix3x3* matrix) {
    set_sin_cos(amount);

    m11 = matrix->m11; m21 = matrix->m21; m31 = matrix->m31;
    m12 = matrix->m12; m22 = matrix->m22; m32 = matrix->m32;
    m13 = matrix->m13; m23 = matrix->m23; m33 = matrix->m33;

    matrix->m12 = m12 * trig.cos + m13 * trig.sin; 
    matrix->m22 = m22 * trig.cos + m23 * trig.sin; 
    matrix->m32 = m32 * trig.cos + m33 * trig.sin;

    matrix->m13 = m13 * trig.cos - m12 * trig.sin;
    matrix->m23 = m23 * trig.cos - m22 * trig.sin;
    matrix->m33 = m33 * trig.cos - m32 * trig.sin;
};

inline void relative_roll(f32 amount, Matrix3x3* matrix) {
    set_sin_cos(amount);

    m11 = matrix->m11; m21 = matrix->m21; m31 = matrix->m31;
    m12 = matrix->m12; m22 = matrix->m22; m32 = matrix->m32;
    m13 = matrix->m13; m23 = matrix->m23; m33 = matrix->m33;

    matrix->m11 = m11 * trig.cos + m12 * trig.sin; 
    matrix->m21 = m21 * trig.cos + m22 * trig.sin; 
    matrix->m31 = m31 * trig.cos + m32 * trig.sin; 

    matrix->m12 = m12 * trig.cos - m11 * trig.sin;
    matrix->m22 = m22 * trig.cos - m21 * trig.sin;
    matrix->m32 = m32 * trig.cos - m31 * trig.sin;
};

inline void yaw(f32 amount) {
    set_sin_cos(amount);

    m11 = yaw_matrix.m11; m21 = yaw_matrix.m21; m31 = yaw_matrix.m31;
    m12 = yaw_matrix.m12; m22 = yaw_matrix.m22; m32 = yaw_matrix.m32;
    m13 = yaw_matrix.m13; m23 = yaw_matrix.m23; m33 = yaw_matrix.m33;

    yaw_matrix.m11 = m11 * trig.cos - m13 * trig.sin; 
    yaw_matrix.m21 = m21 * trig.cos - m23 * trig.sin; 
    yaw_matrix.m31 = m31 * trig.cos - m33 * trig.sin; 

    yaw_matrix.m13 = m13 * trig.cos + m11 * trig.sin;
    yaw_matrix.m23 = m23 * trig.cos + m21 * trig.sin;
    yaw_matrix.m33 = m33 * trig.cos + m31 * trig.sin;
};

inline void pitch(f32 amount) {
    set_sin_cos(amount);

    m11 = pitch_matrix.m11; m21 = pitch_matrix.m21; m31 = pitch_matrix.m31;
    m12 = pitch_matrix.m12; m22 = pitch_matrix.m22; m32 = pitch_matrix.m32;
    m13 = pitch_matrix.m13; m23 = pitch_matrix.m23; m33 = pitch_matrix.m33;

    pitch_matrix.m12 = m12 * trig.cos + m13 * trig.sin; 
    pitch_matrix.m22 = m22 * trig.cos + m23 * trig.sin; 
    pitch_matrix.m32 = m32 * trig.cos + m33 * trig.sin;

    pitch_matrix.m13 = m13 * trig.cos - m12 * trig.sin;
    pitch_matrix.m23 = m23 * trig.cos - m22 * trig.sin;
    pitch_matrix.m33 = m33 * trig.cos - m32 * trig.sin;
};

inline void roll(f32 amount) {
    set_sin_cos(amount);

    m11 = roll_matrix.m11; m21 = roll_matrix.m21; m31 = roll_matrix.m31;
    m12 = roll_matrix.m12; m22 = roll_matrix.m22; m32 = roll_matrix.m32;
    m13 = roll_matrix.m13; m23 = roll_matrix.m23; m33 = roll_matrix.m33;

    roll_matrix.m11 = m11 * trig.cos + m12 * trig.sin; 
    roll_matrix.m21 = m21 * trig.cos + m22 * trig.sin; 
    roll_matrix.m31 = m31 * trig.cos + m32 * trig.sin; 

    roll_matrix.m12 = m12 * trig.cos - m11 * trig.sin;
    roll_matrix.m22 = m22 * trig.cos - m21 * trig.sin;
    roll_matrix.m32 = m32 * trig.cos - m31 * trig.sin;
};

inline void set_yaw(f32 amount) {
    set_sin_cos(amount);

    yaw_matrix.m11 = yaw_matrix.m33 = trig.cos;
    yaw_matrix.m13 = trig.sin;
    yaw_matrix.m31 = -trig.sin;
};

inline void set_pitch(f32 amount) {
    set_sin_cos(amount);

    pitch_matrix.m33 = pitch_matrix.m22 = trig.cos;
    pitch_matrix.m23 = -trig.sin;
    pitch_matrix.m32 = trig.sin;
};

inline void set_roll(f32 amount) {
    set_sin_cos(amount);

    roll_matrix.m11 = roll_matrix.m22 = trig.cos;
    roll_matrix.m12 = -trig.sin;
    roll_matrix.m21 = trig.sin;
};

void relative_rotate(Matrix3x3* matrix, f32 yaw_amount, f32 pitch_amount, f32 roll_amount) {
    if (yaw_amount) relative_yaw(yaw_amount, matrix);
    if (pitch_amount) relative_pitch(pitch_amount, matrix);
    if (roll_amount) relative_roll(roll_amount, matrix);
}

void absolute_rotate(Matrix3x3* matrix, f32 yaw_amount, f32 pitch_amount, f32 roll_amount) {
    if (roll_amount) set_roll(roll_amount);
    if (pitch_amount) set_pitch(pitch_amount);
    if (yaw_amount) set_yaw(yaw_amount);
    
    matmul(&roll_matrix, &pitch_matrix, &rotation_matrix);
    imatmul(&rotation_matrix, &yaw_matrix);    
    imatmul(matrix, &rotation_matrix);
}

void rotate(Matrix3x3* matrix, f32 yaw_amount, f32 pitch_amount, f32 roll_amount) {
    if (roll_amount) roll(roll_amount);
    if (pitch_amount) pitch(pitch_amount);
    if (yaw_amount) yaw(yaw_amount);
    
    matmul(&roll_matrix, &pitch_matrix, matrix);
    imatmul(matrix, &yaw_matrix);    
    imatmul(matrix, &rotation_matrix);
}

void init_math() {
    setMatrixToIdentity(&yaw_matrix);
    setMatrixToIdentity(&pitch_matrix);
    setMatrixToIdentity(&roll_matrix);
    setMatrixToIdentity(&rotation_matrix);
}

/*
typedef struct Matrix3x3 {Vector3 i, j, k;} Matrix3x3;

static Matrix3x3 camera_orientation = {
        {1,0,0},
        {0,1,0},
        {0,0,1}
};

static Matrix3x3 x_rotation_matrix = {
        {1,0,0},
        {0,1,0},
        {0,0,1}
};

static Matrix3x3 y_rotation_matrix = {
        {1,0,0},
        {0,1,0},
        {0,0,1}
};

void rotate_camera(f32 horizontal_amount, f32 vertical_amount) {
    set_point_on_unit_circle(vertical_amount);
    x_rotation_matrix.k.z = x_rotation_matrix.j.y = cx;
    x_rotation_matrix.k.y = cy;
    x_rotation_matrix.j.z = -cy;

    set_point_on_unit_circle(horizontal_amount);
    y_rotation_matrix.i.x = y_rotation_matrix.k. = cx;
    y_rotation_matrix.k.y = cy;
    y_rotation_matrix.j.z = -cy;


}*/

/*
static f32 x, y, z, r2, i2, j2, k2, ij, jk, ik, rk, rj, ri;
void rotate_vector_by_quaternion(Vector3* v, f32 r, f32 i,  f32 j, f32 k) {
    x = v->x;
    y = v->y;
    z = v->z;

    r2 = r * r;
    i2 = i * i;
    j2 = j * j;
    k2 = k * k;

    ij = i * j;
    ik = i * k;
    jk = j * k;

    ri = r * i;
    rj = r * j;
    rk = r * k;

    v->x = 2*(rj*z + ik*z - rk*y + ij*y) + x*(r2 + i2 - j2 - k2);
    v->y = 2*(rk*x + ij*x - ri*z + jk*z) + y*(r2 - i2 + j2 - k2);
    v->z = 2*(ri*y - rj*x + ik*x + jk*y) + z*(r2 - i2 - j2 + k2);
}

static Vector3 temp_v1, temp_v2;
void rotate_vector_by_quaternion(Vector3* v, Vector3* u, f32 s, Vector3* out) {
    scale(u, 2.0f * dot(u, v), &temp_v1);
    scale(v, s * s - dot(u, u), &temp_v2);
    iadd(&temp_v1, &temp_v2);

    cross(u, v, out);
    iscale(out, 2.0f * s);
    iadd(out, &temp_v1);
}
static f32 cx, cy, t2;
void set_point_on_unit_circle(f32 t) {
    t2 = t * t;
    cx = 1 - t2;
    cy = 2 * t;
    t2 += 1;
    t2 = 1 / t2;
    cx *= t2;
    cy *= t2;
}*/
