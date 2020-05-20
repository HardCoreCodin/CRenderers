#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"

#define SPHERE_RADIUS 1
#define SPHERE_HCOUNT 3
#define SPHERE_VCOUNT 3

void initScene(Scene* scene) {
    scene->camera = Alloc(Camera);
    initCamera(scene->camera);

    scene->sphere_count = SPHERE_HCOUNT * SPHERE_VCOUNT;
    scene->spheres = AllocN(Sphere, scene->sphere_count);

    Sphere* sphere = scene->spheres;

    u8 gap = SPHERE_RADIUS * 3;
    u8 sphere_x = 0;
    u8 sphere_z = 0;

    for (u8 z = 0; z < SPHERE_VCOUNT; z++) {
        sphere_x = 0;

        for (u8 x = 0; x < SPHERE_HCOUNT; x++) {
            sphere->radius = 1;

            sphere->world_position = Alloc(Vector3);
            sphere->view_position = Alloc(Vector3);

            sphere->world_position->x = sphere_x;
            sphere->world_position->y = 0;
            sphere->world_position->z = sphere_z;

            sphere_x += gap;
            sphere++;
        }

        sphere_z += gap;
    }
}