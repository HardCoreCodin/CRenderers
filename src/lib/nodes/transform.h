#pragma once

#include "lib/core/types.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"
#include "lib/memory/allocators.h"

void initTransform2D(Transform2D* transform) {
    initMatrix2x2(&transform->rotation);
    setMatrix2x2ToIdentity(transform->rotation);
    transform->right = transform->rotation.x_axis;
    transform->forward = transform->rotation.y_axis;
    transform->position = Alloc(Vector2);
}

void initTransform3D(Transform3D* transform) {
    initMatrix3x3(&transform->rotation);
    initMatrix3x3(&transform->yaw);
    initMatrix3x3(&transform->pitch);
    initMatrix3x3(&transform->roll);

    setMatrix3x3ToIdentity(transform->yaw);
    setMatrix3x3ToIdentity(transform->pitch);
    setMatrix3x3ToIdentity(transform->roll);
    setMatrix3x3ToIdentity(transform->rotation);

    transform->position = (Vector3*)allocate(sizeof(Vector3));

    transform->up = transform->rotation.y_axis;
    transform->right = transform->rotation.x_axis;
    transform->forward = transform->rotation.z_axis;
}
