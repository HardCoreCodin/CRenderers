#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/nodes/transform.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"

Camera* createCamera() {
    Camera* camera = Alloc(Camera);
    camera->focal_length = 2;
    Transform3D* tr3 = camera->transform = createTransform3D();
    Transform2D* tr2 = camera->transform2D = createTransform2D();
    tr2->rotation->x_axis = tr2->right = (Vector2*)tr3->rotation->x_axis;
    tr2->rotation->y_axis = tr2->forward = (Vector2*)tr3->rotation->y_axis;
    tr2->position = (Vector2*)camera->transform->position;

    return camera;
}