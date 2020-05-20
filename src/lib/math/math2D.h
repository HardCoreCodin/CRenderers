#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"

void setMatrix2x2ToIdentity(Matrix2x2* M) {
    M->x_axis->x = M->y_axis->y = 1.0f;
    M->x_axis->y = M->y_axis->x = 0.0f;
}

Matrix2x2* createMatrix2x2() {
    Matrix2x2* matrix = Alloc(Matrix2x2);
    matrix->x_axis = Alloc(Vector2);
    matrix->y_axis = Alloc(Vector2);
    setMatrix2x2ToIdentity(matrix);
    return matrix;
}

void getPointOnUnitCircle(f32 t, f32* s, f32*c) {
    f32 t_squared = t * t;
    f32 factor = 1 / (1 + t_squared);
    *c = factor - factor * t_squared;
    *s = factor * 2 * t;
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
    out->x = in->x*matrix->x_axis->x + in->y*matrix->y_axis->x;
    out->y = in->x*matrix->x_axis->y + in->y*matrix->y_axis->y;
}
inline void imul2D(Vector2* v, Matrix2x2* matrix) {
    f32 x = v->x;
    f32 y = v->y;

    v->x = x*matrix->x_axis->x + y*matrix->y_axis->x;
    v->y = x*matrix->x_axis->y + y*matrix->y_axis->y;
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

inline void matmul2D(Matrix2x2* a, Matrix2x2* b, Matrix2x2* out) {
    out->x_axis->x = a->x_axis->x*b->x_axis->x + a->x_axis->y*b->y_axis->x; // Row 1 | Column 1
    out->x_axis->y = a->x_axis->x*b->x_axis->y + a->x_axis->y*b->y_axis->y; // Row 1 | Column 2
    
    out->y_axis->x = a->y_axis->x*b->x_axis->x + a->y_axis->y*b->y_axis->x; // Row 2 | Column 1
    out->y_axis->y = a->y_axis->x*b->x_axis->y + a->y_axis->y*b->y_axis->y; // Row 2 | Column 2
}

inline void imatmul2D(Matrix2x2* a, Matrix2x2* b) {
    f32 m11 = a->x_axis->x; f32 m21 = a->y_axis->x;
    f32 m12 = a->x_axis->y; f32 m22 = a->y_axis->y;

    a->x_axis->x = m11*b->x_axis->x + m12*b->y_axis->x; // Row 1 | Column 1
    a->x_axis->y = m11*b->x_axis->y + m12*b->y_axis->y; // Row 1 | Column 2

    a->y_axis->x = m21*b->x_axis->x + m22*b->y_axis->x; // Row 2 | Column 1
    a->y_axis->y = m21*b->x_axis->y + m22*b->y_axis->y; // Row 2 | Column 2
}

inline void rotateMatrix2D(Matrix2x2* matrix, f32 amount) {
    f32 s, c;
    getPointOnUnitCircle(amount, &s, &c);
    
    Vector2 x_axis = *matrix->x_axis;
    Vector2 y_axis = *matrix->y_axis;

    matrix->x_axis->x = x_axis.x*c + x_axis.y*s;
    matrix->y_axis->x = y_axis.x*c + y_axis.y*s;

    matrix->x_axis->y = x_axis.y*c - x_axis.x*s;
    matrix->y_axis->y = y_axis.y*c - y_axis.x*s;
};

inline void setRotation2D(f32 amount, Matrix2x2* matrix) {
    f32 s, c;
    getPointOnUnitCircle(amount, &s, &c);

    matrix->x_axis->x = matrix->y_axis->y = c;
    matrix->x_axis->y = -s;
    matrix->y_axis->x = +s;
};

inline void rotate2D(f32 amount, Matrix2x2* matrix) {
    setRotation2D(amount, matrix);
    imatmul2D(matrix, matrix);
};