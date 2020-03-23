#pragma once

#include "lib/core/types.h"
#include "lib/core/memory.h"
#include "lib/math/math3D.h"

typedef struct Transform3D {
    Matrix3x3* yaw;
    Matrix3x3* pitch;
    Matrix3x3* roll;

    Matrix3x3* rotation;

    Vector3* forward;
    Vector3* right;
    Vector3* up;
} Transform3D;

void initTransform3D(Transform3D* transform, Memory* memory) {
    transform->yaw = (Matrix3x3*)allocate(memory, sizeof(Matrix3x3));
    transform->pitch = (Matrix3x3*)allocate(memory, sizeof(Matrix3x3));
    transform->roll = (Matrix3x3*)allocate(memory, sizeof(Matrix3x3));

    transform->rotation = (Matrix3x3*)allocate(memory, sizeof(Matrix3x3));

    setMatrix3x3ToIdentity(transform->yaw);
    setMatrix3x3ToIdentity(transform->pitch);
    setMatrix3x3ToIdentity(transform->roll);

    setMatrix3x3ToIdentity(transform->rotation);

    transform->up = &transform->rotation->j;
    transform->right = &transform->rotation->i;
    transform->forward = &transform->rotation->k;
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

typedef struct Transform2D {
    Matrix2x2* rotation;
    Vector2* forward;
    Vector2* right;
} Transform2D;

void initTransform2D(Transform2D* transform, Memory* memory) {
    transform->rotation = (Matrix2x2*)allocate(memory, sizeof(Matrix2x2));

    setMatrix2x2ToIdentity(transform->rotation);

    transform->right = &transform->rotation->i;
    transform->forward = &transform->rotation->j;
}
