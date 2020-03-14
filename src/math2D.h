#pragma once
#pragma warning(disable : 4201)
#include "types.h"

typedef struct Vector2 {
    f32 x, y;
} Vector2;

typedef union {
    struct {
        f32 m11, m12,
            m21, m22;
    };
    struct {
        Vector2 i, j;
    };
} Matrix2x2;

inline void sub2D(Vector2* p1, Vector2* p2, Vector2* out) {
    out->x = p1->x - p2->x;
    out->y = p1->y - p2->y;
}
inline void isub(Vector2* p1, Vector2* p2) {
    p1->x -= p2->x;
    p1->y -= p2->y;
}
inline void add2D(Vector2* p1, Vector2* p2, Vector2* out) {
    out->x = p1->x + p2->x;
    out->y = p1->y + p2->y;
}
inline void iadd2D(Vector2* p1, Vector2* p2) {
    p1->x += p2->x;
    p1->y += p2->y;
}
inline void scale2D(Vector2* p1, f32 factor, Vector2* out) {
    out->x = p1->x * factor;
    out->y = p1->y * factor;
}
inline void iscale2D(Vector2* v, f32 factor) {
    v->x *= factor;
    v->y *= factor;
}
inline void idiv2D(Vector2* v, f32 factor) {
    v->x /= factor;
    v->y /= factor;
}
inline f32 dot2D(Vector2* p1, Vector2* p2) {
    return (
        (p1->x * p2->x) +
        (p1->y * p2->y)
    );
}
inline f32 squaredLength2D(Vector2* v) {
    return (
        (v->x * v->x) +
        (v->y * v->y)
    );
}

void setMatrix2x2ToIdentity(Matrix2x2* M) {
    M->m11 = M->m22 = 1.0f; 
    M->m12 = M->m21 = 0.0f;
}

inline void matmul2D(Matrix2x2* a, Matrix2x2* b, Matrix2x2* out) {
    out->m11 = a->m11*b->m11 + a->m12*b->m21; // Row 1 | Column 1
    out->m12 = a->m11*b->m12 + a->m12*b->m22; // Row 1 | Column 2
    
    out->m21 = a->m21*b->m11 + a->m22*b->m21; // Row 2 | Column 1
    out->m22 = a->m21*b->m12 + a->m22*b->m22; // Row 2 | Column 2
}

inline void imatmul2D(Matrix2x2* a, Matrix2x2* b) {
    f32 m11 = a->m11; f32 m21 = a->m21;
    f32 m12 = a->m12; f32 m22 = a->m22;

    a->m11 = m11*b->m11 + m12*b->m21; // Row 1 | Column 1
    a->m12 = m11*b->m12 + m12*b->m22; // Row 1 | Column 2

    a->m21 = m21*b->m11 + m22*b->m21; // Row 2 | Column 1
    a->m22 = m21*b->m12 + m22*b->m22; // Row 2 | Column 2
}

inline void rotateMatrix2D(Matrix2x2* matrix, f32 amount) {
    set_point_on_unit_circle(amount);

    f32 m11 = matrix->m11; f32 m21 = matrix->m21;
    f32 m12 = matrix->m12; f32 m22 = matrix->m22;

    matrix->m11 = c*m11 + s*m12; 
    matrix->m21 = c*m21 + s*m22; 

    matrix->m12 = c*m12 - s*m11;
    matrix->m22 = c*m22 - s*m21;
};

inline void setRotation2D(Matrix2x2* matrix, f32 amount) {
    set_point_on_unit_circle(amount);

    matrix->m11 = matrix->m22 = c;
    matrix->m12 = -s;
    matrix->m21 = +s;
};

void init_math2D() {

}