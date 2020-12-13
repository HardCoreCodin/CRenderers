#pragma once

#include "lib/core/types.h"
#include "lib/globals/camera.h"
#include "lib/memory/allocators.h"
#include "lib/nodes/transform.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"

void initCamera(Camera* camera) {
    camera->focal_length = 2;
    initXform3(&camera->transform);
}