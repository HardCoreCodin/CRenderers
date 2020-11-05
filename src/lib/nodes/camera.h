#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/nodes/transform.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"

Camera* createCamera() {
    Camera* camera = Alloc(Camera);
    camera->focal_length = 2;
    camera->one_over_focal_length = 1.0f / camera->focal_length;
    initXform3(&camera->transform);
    return camera;
}