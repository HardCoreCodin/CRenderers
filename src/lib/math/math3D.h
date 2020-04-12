#pragma once

#include "lib/core/types.h"
#include "math2D.h"
#include "lib/memory/allocators.h"

void initMatrix3x3(Matrix3x3* matrix) {
    matrix->x_axis = Alloc(Vector3);
    matrix->y_axis = Alloc(Vector3);
    matrix->z_axis = Alloc(Vector3);
}

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
inline void mul3D(Vector3* in, Matrix3x3 matrix, Vector3* out) {
    out->x = in->x*matrix.x_axis->x + in->y*matrix.y_axis->x + in->z*matrix.z_axis->x;
    out->y = in->x*matrix.x_axis->y + in->y*matrix.y_axis->y + in->z*matrix.z_axis->y;
    out->z = in->x*matrix.x_axis->z + in->y*matrix.y_axis->z + in->z*matrix.z_axis->z;    
}
inline void imul3D(Vector3* v, Matrix3x3 matrix) {
    f32 x = v->x;
    f32 y = v->y;
    f32 z = v->z;

    v->x = x*matrix.x_axis->x + y*matrix.y_axis->x + z*matrix.z_axis->x;
    v->y = x*matrix.x_axis->y + y*matrix.y_axis->y + z*matrix.z_axis->y;
    v->z = x*matrix.x_axis->z + y*matrix.y_axis->z + z*matrix.z_axis->z;    
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

void setMatrix3x3ToIdentity(Matrix3x3 M) {
    M.x_axis->x = M.y_axis->y = M.z_axis->z = 1.0f; 
    M.x_axis->y = M.y_axis->x = M.y_axis->z = M.z_axis->y = M.x_axis->z = M.z_axis->x = 0.0f;
}

void transposeMatrix3D(Matrix3x3 M, Matrix3x3 O) {
    O.x_axis->x = M.x_axis->x; O.y_axis->y = M.y_axis->y; O.z_axis->z = M.z_axis->z;
    O.x_axis->y = M.y_axis->x; O.y_axis->x = M.x_axis->y;
    O.x_axis->z = M.z_axis->x; O.z_axis->x = M.x_axis->z;
    O.y_axis->z = M.z_axis->y; O.z_axis->y = M.y_axis->z;
}

inline void matMul3D(Matrix3x3 a, Matrix3x3 b, Matrix3x3 out) {
    out.x_axis->x = a.x_axis->x*b.x_axis->x + a.x_axis->y*b.y_axis->x + a.x_axis->z*b.z_axis->x; // Row 1 | Column 1
    out.x_axis->y = a.x_axis->x*b.x_axis->y + a.x_axis->y*b.y_axis->y + a.x_axis->z*b.z_axis->y; // Row 1 | Column 2
    out.x_axis->z = a.x_axis->x*b.x_axis->z + a.x_axis->y*b.y_axis->z + a.x_axis->z*b.z_axis->z; // Row 1 | Column 3

    out.y_axis->x = a.y_axis->x*b.x_axis->x + a.y_axis->y*b.y_axis->x + a.y_axis->z*b.z_axis->x; // Row 2 | Column 1
    out.y_axis->y = a.y_axis->x*b.x_axis->y + a.y_axis->y*b.y_axis->y + a.y_axis->z*b.z_axis->y; // Row 2 | Column 2
    out.y_axis->z = a.y_axis->x*b.x_axis->z + a.y_axis->y*b.y_axis->z + a.y_axis->z*b.z_axis->z; // Row 2 | Column 3

    out.z_axis->x = a.z_axis->x*b.x_axis->x + a.z_axis->y*b.y_axis->x + a.z_axis->z*b.z_axis->x; // Row 3 | Column 1
    out.z_axis->y = a.z_axis->x*b.x_axis->y + a.z_axis->y*b.y_axis->y + a.z_axis->z*b.z_axis->y; // Row 3 | Column 2
    out.z_axis->z = a.z_axis->x*b.x_axis->z + a.z_axis->y*b.y_axis->z + a.z_axis->z*b.z_axis->z; // Row 3 | Column 3
}

inline void imatMul3D(Matrix3x3 a, Matrix3x3 b) {
    Vector3 x_axis = *a.x_axis;
    Vector3 y_axis = *a.y_axis;
    Vector3 z_axis = *a.z_axis;

    a.x_axis->x = x_axis.x*b.x_axis->x + x_axis.y*b.y_axis->x + x_axis.z*b.z_axis->x; // Row 1 | Column 1
    a.x_axis->y = x_axis.x*b.x_axis->y + x_axis.y*b.y_axis->y + x_axis.z*b.z_axis->y; // Row 1 | Column 2
    a.x_axis->z = x_axis.x*b.x_axis->z + x_axis.y*b.y_axis->z + x_axis.z*b.z_axis->z; // Row 1 | Column 3

    a.y_axis->x = y_axis.x*b.x_axis->x + y_axis.y*b.y_axis->x + y_axis.z*b.z_axis->x; // Row 2 | Column 1
    a.y_axis->y = y_axis.x*b.x_axis->y + y_axis.y*b.y_axis->y + y_axis.z*b.z_axis->y; // Row 2 | Column 2
    a.y_axis->z = y_axis.x*b.x_axis->z + y_axis.y*b.y_axis->z + y_axis.z*b.z_axis->z; // Row 2 | Column 3

    a.z_axis->x = z_axis.x*b.x_axis->x + z_axis.y*b.y_axis->x + z_axis.z*b.z_axis->x; // Row 3 | Column 1
    a.z_axis->y = z_axis.x*b.x_axis->y + z_axis.y*b.y_axis->y + z_axis.z*b.z_axis->y; // Row 3 | Column 2
    a.z_axis->z = z_axis.x*b.x_axis->z + z_axis.y*b.y_axis->z + z_axis.z*b.z_axis->z; // Row 3 | Column 3
}

inline void relativeYaw3D(f32 yaw, Matrix3x3 yaw_matrix) {
    f32 s, c;
    getPointOnUnitCircle(yaw, &s, &c);

    Vector3 x_axis = *yaw_matrix.x_axis;
    Vector3 y_axis = *yaw_matrix.y_axis;
    Vector3 z_axis = *yaw_matrix.z_axis;

    yaw_matrix.x_axis->x = c * x_axis.x - s * x_axis.z;
    yaw_matrix.y_axis->x = c * y_axis.x - s * y_axis.z;
    yaw_matrix.z_axis->x = c * z_axis.x - s * z_axis.z;

    yaw_matrix.x_axis->z = c * x_axis.z + s * x_axis.x;
    yaw_matrix.y_axis->z = c * y_axis.z + s * y_axis.x;
    yaw_matrix.z_axis->z = c * z_axis.z + s * z_axis.x;
};

inline void relativePitch3D(f32 pitch, Matrix3x3 pitch_matrix) {
    f32 s, c;
    getPointOnUnitCircle(pitch, &s, &c);

    Vector3 x_axis = *pitch_matrix.x_axis;
    Vector3 y_axis = *pitch_matrix.y_axis;
    Vector3 z_axis = *pitch_matrix.z_axis;

    pitch_matrix.x_axis->y = c * x_axis.y + s * x_axis.z;
    pitch_matrix.y_axis->y = c * y_axis.y + s * y_axis.z;
    pitch_matrix.z_axis->y = c * z_axis.y + s * z_axis.z;

    pitch_matrix.x_axis->z = c * x_axis.z - s * x_axis.y;
    pitch_matrix.y_axis->z = c * y_axis.z - s * y_axis.y;
    pitch_matrix.z_axis->z = c * z_axis.z - s * z_axis.y;
};

inline void relativeRoll3D(f32 roll, Matrix3x3 roll_matrix) {
    f32 s, c;
    getPointOnUnitCircle(roll, &s, &c);

    Vector3 x_axis = *roll_matrix.x_axis;
    Vector3 y_axis = *roll_matrix.y_axis;
    Vector3 z_axis = *roll_matrix.z_axis;

    roll_matrix.x_axis->x = c * x_axis.x + s * x_axis.y;
    roll_matrix.y_axis->x = c * y_axis.x + s * y_axis.y;
    roll_matrix.z_axis->x = c * z_axis.x + s * z_axis.y;

    roll_matrix.x_axis->y = c * x_axis.y - s * x_axis.x;
    roll_matrix.y_axis->y = c * y_axis.y - s * y_axis.x;
    roll_matrix.z_axis->y = c * z_axis.y - s * z_axis.x;
};

inline void yaw3D(f32 yaw, Matrix3x3 yaw_matrix) {
    f32 s, c;
    getPointOnUnitCircle(yaw, &s, &c);

    Vector3 x_axis = *yaw_matrix.x_axis;
    Vector3 y_axis = *yaw_matrix.y_axis;
    Vector3 z_axis = *yaw_matrix.z_axis;

    yaw_matrix.x_axis->x = c * x_axis.x - s * x_axis.z;
    yaw_matrix.y_axis->x = c * y_axis.x - s * y_axis.z;
    yaw_matrix.z_axis->x = c * z_axis.x - s * z_axis.z;

    yaw_matrix.x_axis->z = c * x_axis.z + s * x_axis.x;
    yaw_matrix.y_axis->z = c * y_axis.z + s * y_axis.x;
    yaw_matrix.z_axis->z = c * z_axis.z + s * z_axis.x;
};

inline void pitch3D(f32 pitch, Matrix3x3 pitch_matrix) {
    f32 s, c;
    getPointOnUnitCircle(pitch, &s, &c);

    Vector3 x_axis = *pitch_matrix.x_axis;
    Vector3 y_axis = *pitch_matrix.y_axis;
    Vector3 z_axis = *pitch_matrix.z_axis;

    pitch_matrix.x_axis->y = c * x_axis.y + s * x_axis.z;
    pitch_matrix.y_axis->y = c * y_axis.y + s * y_axis.z;
    pitch_matrix.z_axis->y = c * z_axis.y + s * z_axis.z;

    pitch_matrix.x_axis->z = c * x_axis.z - s * x_axis.y;
    pitch_matrix.y_axis->z = c * y_axis.z - s * y_axis.y;
    pitch_matrix.z_axis->z = c * z_axis.z - s * z_axis.y;
};

inline void roll3D(f32 roll, Matrix3x3 roll_matrix) {
    f32 s, c;
    getPointOnUnitCircle(roll, &s, &c);

    Vector3 x_axis = *roll_matrix.x_axis;
    Vector3 y_axis = *roll_matrix.y_axis;
    Vector3 z_axis = *roll_matrix.z_axis;
    
    roll_matrix.x_axis->x = c * x_axis.x + s * x_axis.y;
    roll_matrix.y_axis->x = c * y_axis.x + s * y_axis.y;
    roll_matrix.z_axis->x = c * z_axis.x + s * z_axis.y;

    roll_matrix.x_axis->y = c * x_axis.y - s * x_axis.x;
    roll_matrix.y_axis->y = c * y_axis.y - s * y_axis.x;
    roll_matrix.z_axis->y = c * z_axis.y - s * z_axis.x;
};

inline void setYaw3D(f32 yaw, Matrix3x3 yaw_matrix) {
    f32 s, c;
    getPointOnUnitCircle(yaw, &s, &c);

    yaw_matrix.x_axis->x = yaw_matrix.z_axis->z = c;
    yaw_matrix.x_axis->z = +s;
    yaw_matrix.z_axis->x = -s;
};

inline void setPitch3D(f32 pitch, Matrix3x3 pitch_matrix) {
    f32 s, c;
    getPointOnUnitCircle(pitch, &s, &c);

    pitch_matrix.z_axis->z = pitch_matrix.y_axis->y = c;
    pitch_matrix.y_axis->z = -s;
    pitch_matrix.z_axis->y = +s;
};

inline void setRoll3D(f32 roll, Matrix3x3 roll_matrix) {
    f32 s, c;
    getPointOnUnitCircle(roll, &s, &c);

    roll_matrix.x_axis->x = roll_matrix.y_axis->y = c;
    roll_matrix.x_axis->y = -s;
    roll_matrix.y_axis->x = +s;
};

void rotateRelative3D(f32 yaw, f32 pitch, f32 roll, Matrix3x3 rotation_matrix) {
    if (yaw) relativeYaw3D(yaw, rotation_matrix);
    if (pitch) relativePitch3D(pitch, rotation_matrix);
    if (roll) relativeRoll3D(roll, rotation_matrix);
}

void rotateAbsolute3D(
        f32 yaw,
        f32 pitch,
        f32 roll,

        Matrix3x3 yaw_matrix,
        Matrix3x3 pitch_matrix,
        Matrix3x3 roll_matrix,

        Matrix3x3 rotation_matrix
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
