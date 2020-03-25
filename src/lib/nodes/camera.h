#pragma once

#include "lib/core/types.h"
#include "lib/nodes/transform.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"

typedef struct Camera3D {
    f32 focal_length;
    Vector3* position;
    Transform3D* transform;
} Camera3D;

void initCamera3D(Camera3D* camera) {
    camera->focal_length = 1;
    camera->position = (Vector3*)allocate(sizeof(Vector3));
    camera->transform = (Transform3D*)allocate(sizeof(Transform3D));
    initTransform3D(camera->transform);
}

typedef struct Camera2D {
    f32 focal_length;
    Vector2* position;
    Transform2D* transform;
} Camera2D;

void initCamera2D(Camera2D* camera) {
    camera->focal_length = 1;
    camera->position = (Vector2*)allocate(sizeof(Vector2));
    initTransform2D(camera->transform);
}
