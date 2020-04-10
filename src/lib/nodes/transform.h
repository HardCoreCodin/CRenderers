#pragma once

#include "lib/core/types.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"
#include "lib/memory/allocators.h"

void initTransform2D(Transform2D* transform) {
    transform->rotation = (Matrix2x2*)allocate(sizeof(Matrix2x2));
    transform->position = (Vector2*)allocate(sizeof(Vector2));
    setMatrix2x2ToIdentity(transform->rotation);

    transform->right = &transform->rotation->i;
    transform->forward = &transform->rotation->j;
}

void initTransform3D(Transform3D* transform) {
    transform->yaw = (Matrix3x3*)allocate(sizeof(Matrix3x3));
    transform->pitch = (Matrix3x3*)allocate(sizeof(Matrix3x3));
    transform->roll = (Matrix3x3*)allocate(sizeof(Matrix3x3));

    transform->rotation = (Matrix3x3*)allocate(sizeof(Matrix3x3));
    transform->rotation_inverted = (Matrix3x3*)allocate(sizeof(Matrix3x3));
    transform->position = (Vector3*)allocate(sizeof(Vector3));

    setMatrix3x3ToIdentity(transform->yaw);
    setMatrix3x3ToIdentity(transform->pitch);
    setMatrix3x3ToIdentity(transform->roll);

    setMatrix3x3ToIdentity(transform->rotation);
    setMatrix3x3ToIdentity(transform->rotation_inverted);

    transform->up = &transform->rotation->j;
    transform->right = &transform->rotation->i;
    transform->forward = &transform->rotation->k;
}
