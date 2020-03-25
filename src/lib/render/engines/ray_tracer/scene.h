#pragma once

#include "lib/core/types.h"
#include "lib/core/memory.h"
#include "lib/math/math3D.h"

#define SPHERE_RADIUS 1
#define SPHERE_HCOUNT 2
#define SPHERE_VCOUNT 2

typedef struct Sphere {
    f32 radius;
    Vector3 position;
} Sphere;

typedef struct Scene {
    u8 sphere_count;
    Sphere* spheres;
} Scene;

static Scene scene;

void initScene() {
    scene.sphere_count = SPHERE_HCOUNT * SPHERE_VCOUNT;
    scene.spheres = (Sphere*)allocate(sizeof(Sphere) * scene.sphere_count);

    Sphere* sphere = scene.spheres;

    u8 gap = SPHERE_RADIUS * 3;
    u8 sphere_x = 0;
    u8 sphere_z = 0;

    for (u8 z = 0; z < SPHERE_VCOUNT; z++) {
        sphere_x = 0;

        for (u8 x = 0; x < SPHERE_HCOUNT; x++) {
            sphere->radius = 1;

            sphere->position.x = sphere_x;
            sphere->position.y = 0;
            sphere->position.z = sphere_z;

            sphere_x += gap;
            sphere++;
        }

        sphere_z += gap;
    }

}