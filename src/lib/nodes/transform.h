#pragma once

#include "lib/core/types.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"
#include "lib/memory/allocators.h"

Transform2D* createTransform2D() {
    Transform2D* transform = Alloc(Transform2D);
    transform->rotation = createMatrix2x2();
    transform->right = transform->rotation->x_axis;
    transform->forward = transform->rotation->y_axis;
    transform->position = Alloc(Vector2);

    return transform;
}

Transform3D* createTransform3D() {
    Transform3D* transform = Alloc(Transform3D);

    transform->rotation = createMatrix3x3();
    transform->yaw = createMatrix3x3();
    transform->pitch = createMatrix3x3();
    transform->roll = createMatrix3x3();
    transform->position = Alloc(Vector3);
    transform->up = transform->rotation->y_axis;
    transform->right = transform->rotation->x_axis;
    transform->forward = transform->rotation->z_axis;

    return transform;
}
