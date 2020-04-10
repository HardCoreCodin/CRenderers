#pragma once

#include "lib/core/types.h"

Vector2 vec2;
Matrix2x2 mat2;

void setPointOnUnitCircle(f32 t) {
    f32 t_squared = t * t;
    f32 factor = 1 / (1 + t_squared);
    vec2.x = factor - factor * t_squared;
    vec2.y = factor * 2 * t;
}

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
inline void mul2D(Vector2* in, Matrix2x2* matrix, Vector2* out) {
    out->x = in->x*matrix->m11 + in->y*matrix->m21;
    out->y = in->x*matrix->m12 + in->y*matrix->m22;
}
inline void imul2D(Vector2* v, Matrix2x2* matrix) {
    f32 x = v->x;
    f32 y = v->y;

    v->x = x*matrix->m11 + y*matrix->m21;
    v->y = x*matrix->m12 + y*matrix->m22;
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
    setPointOnUnitCircle(amount);

    mat2.m11 = matrix->m11; mat2.m21 = matrix->m21;
    mat2.m12 = matrix->m12; mat2.m22 = matrix->m22;

    matrix->m11 = vec2.x*mat2.m11 + vec2.y*mat2.m12;
    matrix->m21 = vec2.x*mat2.m21 + vec2.y*mat2.m22;

    matrix->m12 = vec2.x*mat2.m12 - vec2.y*mat2.m11;
    matrix->m22 = vec2.x*mat2.m22 - vec2.y*mat2.m21;
};

inline void setRotation2D(f32 amount, Matrix2x2* matrix) {
    setPointOnUnitCircle(amount);

    matrix->m11 = matrix->m22 = vec2.x;
    matrix->m12 = -vec2.y;
    matrix->m21 = +vec2.y;
};

inline void rotate2D(f32 amount, Matrix2x2* matrix) {
    setRotation2D(amount, &mat2);
    imatmul2D(matrix, &mat2);
};