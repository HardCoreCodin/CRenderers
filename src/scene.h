#pragma once
#include "core.h"

typedef struct Sphere {
    Vector3* position; 
    f32 radius;
} Sphere;

static Sphere sphere;

Vector3* sphere_positions;
u8 sphere_count;

void init_spheres(u8 radius, u8 horizontal_count, u8 vertical_count) {
    sphere_count = horizontal_count * vertical_count;
    sphere_positions = (Vector3*)allocate_memory(sizeof(Vector3) * sphere_count);
    
    sphere.position = sphere_positions;
    sphere.radius = radius;

    u8 gap = radius * 3;
    u8 sphere_x = 0;
    u8 sphere_z = 0;

    for (u8 z = 0; z < vertical_count; z++) {
        sphere_x = 0;

        for (u8 x = 0; x < horizontal_count; x++) {
            sphere.position->x = sphere_x;
            sphere.position->y = 0;
            sphere.position->z = sphere_z;

            sphere_x += gap;
            sphere.position++;
        }

        sphere_z += gap;
    }
}

void init_scene() {
    init_spheres(1, 5, 5);
}
