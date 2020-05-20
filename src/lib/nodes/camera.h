#pragma once

#include "lib/core/types.h"
#include "lib/nodes/transform.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"

void initCamera(Camera* camera) {
    camera->focal_length = 2;
    camera->transform = Alloc(Transform3D);
    camera->transform2D = Alloc(Transform2D);
    initTransform3D(camera->transform);
    initTransform2D(camera->transform2D);

    camera->transform2D->rotation->x_axis = camera->transform2D->right = (Vector2*)camera->transform->rotation->x_axis;
    camera->transform2D->rotation->y_axis = camera->transform2D->forward = (Vector2*)camera->transform->rotation->y_axis;
    camera->transform2D->position = (Vector2*)camera->transform->position;
}