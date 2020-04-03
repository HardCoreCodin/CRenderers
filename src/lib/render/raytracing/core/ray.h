#pragma once

#include "lib/math/math3D.h"
#include "lib/memory/allocators.h"

typedef struct Ray3D {
    Vector3* origin;
    Vector3* direction;
    Vector3* hit_normal;
    Vector3* hit_position;
} Ray3D;

void initRay(Ray3D* ray) {
    ray->hit_normal = (Vector3*)allocate(sizeof(Vector3));
    ray->hit_position = (Vector3*)allocate(sizeof(Vector3));
    fill3D(ray->hit_normal, 0);
    fill3D(ray->hit_position, 0);
}