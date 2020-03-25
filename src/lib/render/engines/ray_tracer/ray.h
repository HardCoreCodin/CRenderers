#pragma once

#include "lib/core/types.h"
#include "lib/core/memory.h"
#include "lib/math/math3D.h"

typedef struct RayHit {
    Vector3 normal;
    Vector3 position;
    f32 distance,
        delta_of_squares,
        origin_to_closest,
        origin_to_closest_minus_delta_of_squares;
} RayHit;

typedef struct Ray {
    Vector3* origin;
    Vector3* direction;

    RayHit* closest_hit;
    RayHit* current_hit;
} Ray;

void initHit(RayHit* hit) {
    hit->distance = 0;
    hit->delta_of_squares = 0;
    hit->origin_to_closest = 0;
    hit->origin_to_closest_minus_delta_of_squares = 0;
    fill3D(&hit->normal, 0);
    fill3D(&hit->position, 0);
};

void initRay(Ray* ray) {
    ray->current_hit = (RayHit*)allocate(sizeof(RayHit));
    ray->closest_hit = (RayHit*)allocate(sizeof(RayHit));

    initHit(ray->current_hit);
    initHit(ray->closest_hit);
}