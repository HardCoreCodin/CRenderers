#pragma once

#include "lib/core/types.h"
#include "lib/nodes/transform.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"

void initCamera2D(Camera2D* camera) {
    camera->focal_length = 1;
    camera->transform = (Transform2D*)allocate(sizeof(Transform2D));
    initTransform2D(camera->transform);
}

void initCamera3D(Camera3D* camera) {
    camera->focal_length = 2;
    camera->transform = (Transform3D*)allocate(sizeof(Transform3D));
    initTransform3D(camera->transform);
}