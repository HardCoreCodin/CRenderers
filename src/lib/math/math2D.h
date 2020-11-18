#pragma once

#include <math.h>
#include "lib/core/types.h"
#include "lib/memory/allocators.h"

inline void setMat2ToIdentity(mat2 *m) {
    m->X.x = 1; m->X.y = 0;
    m->Y.x = 0; m->Y.y = 1;
}

mat2* createMat2() {
    mat2* matrix = Alloc(mat2);
    setMat2ToIdentity(matrix);
    return matrix;
}

vec2 getPointOnUnitCircle(f32 t) {
    f32 t_squared = t * t;
    f32 factor = 1 / (1 + t_squared);
    vec2 xy = {factor - factor * t_squared, factor * 2 * t};
    return xy;
}

inline void fillVec2(vec2* v, f32 value) {
    v->x = v->y = value;
}

inline bool nonZeroVec2(vec2 *v) {
    return v->x != 0 ||
           v->y != 0;
}

inline void subVec2(vec2* p1, vec2* p2, vec2* out) {
    out->x = p1->x - p2->x;
    out->y = p1->y - p2->y;
}
inline void isubVec2(vec2* p1, vec2* p2) {
    p1->x -= p2->x;
    p1->y -= p2->y;
}
inline void addVec2(vec2* p1, vec2* p2, vec2* out) {
    out->x = p1->x + p2->x;
    out->y = p1->y + p2->y;
}
inline void iaddVec2(vec2* p1, vec2* p2) {
    p1->x += p2->x;
    p1->y += p2->y;
}
inline void scaleVec2(vec2* p1, f32 factor, vec2* out) {
    out->x = p1->x * factor;
    out->y = p1->y * factor;
}
inline void iscaleVec2(vec2* v, f32 factor) {
    v->x *= factor;
    v->y *= factor;
}
inline void mulVec2Mat2(vec2* in, mat2* matrix, vec2* out) {
    out->x = in->x*matrix->X.x + in->y*matrix->Y.x;
    out->y = in->x*matrix->X.y + in->y*matrix->Y.y;
}
inline void imulVec2Mat2(vec2* v, mat2* matrix) {
    f32 x = v->x;
    f32 y = v->y;

    v->x = x*matrix->X.x + y*matrix->Y.x;
    v->y = x*matrix->X.y + y*matrix->Y.y;
}
inline f32 dotVec2(vec2* p1, vec2* p2) {
    return (
        (p1->x * p2->x) +
        (p1->y * p2->y)
    );
}
inline f32 squaredLengthVec2(vec2* v) {
    return (
        (v->x * v->x) +
        (v->y * v->y)
    );
}

inline void mulMat2(mat2* a, mat2* b, mat2* out) {
    out->X.x = a->X.x*b->X.x + a->X.y*b->Y.x; // Row 1 | Column 1
    out->X.y = a->X.x*b->X.y + a->X.y*b->Y.y; // Row 1 | Column 2
    
    out->Y.x = a->Y.x*b->X.x + a->Y.y*b->Y.x; // Row 2 | Column 1
    out->Y.y = a->Y.x*b->X.y + a->Y.y*b->Y.y; // Row 2 | Column 2
}

inline void imulMat2(mat2* a, mat2* b) {
    f32 m11 = a->X.x; f32 m21 = a->Y.x;
    f32 m12 = a->X.y; f32 m22 = a->Y.y;

    a->X.x = m11*b->X.x + m12*b->Y.x; // Row 1 | Column 1
    a->X.y = m11*b->X.y + m12*b->Y.y; // Row 1 | Column 2

    a->Y.x = m21*b->X.x + m22*b->Y.x; // Row 2 | Column 1
    a->Y.y = m21*b->X.y + m22*b->Y.y; // Row 2 | Column 2
}

inline void rotateMat2(mat2* matrix, f32 amount) {
    vec2 xy = getPointOnUnitCircle(amount);
    vec2 X = matrix->X;
    vec2 Y = matrix->Y;

    matrix->X.x = X.x*xy.x + X.y*xy.y;
    matrix->Y.x = Y.x*xy.x + Y.y*xy.y;

    matrix->X.y = X.y*xy.x - X.x*xy.y;
    matrix->Y.y = Y.y*xy.x - Y.x*xy.y;
}

inline void setRotationMat2(f32 amount, mat2* matrix) {
    vec2 xy = getPointOnUnitCircle(amount);

    matrix->X.x = matrix->Y.y = xy.x;
    matrix->X.y = -xy.y;
    matrix->Y.x = +xy.y;
}