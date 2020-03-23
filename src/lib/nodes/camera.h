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

void initCamera3D(Camera3D* camera, Memory* memory) {
    camera->focal_length = 1;
    camera->position = (Vector3*)allocate(memory, sizeof(Vector3));
    camera->transform = (Transform3D*)allocate(memory, sizeof(Transform3D));
    initTransform3D(camera->transform, memory);
}

typedef struct Camera2D {
    f32 focal_length;
    Vector2* position;
    Transform2D* transform;
} Camera2D;

void initCamera2D(Camera2D* camera, Memory* memory) {
    camera->focal_length = 1;
    camera->position = (Vector2*)allocate(memory, sizeof(Vector2));
    initTransform2D(camera->transform, memory);
}
