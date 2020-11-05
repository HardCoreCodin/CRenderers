#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"

#define SPHERE_RADIUS 1
#define SPHERE_HCOUNT 3
#define SPHERE_VCOUNT 3
#define LIGHT_COUNT 3
#define PLANE_COUNT 6

Scene scene;

void initScene() {
    scene.camera = createCamera();
    scene.active_sphere_count = 0;
    scene.plane_count = PLANE_COUNT;
    scene.light_count = LIGHT_COUNT;
    scene.sphere_count = SPHERE_HCOUNT * SPHERE_VCOUNT;
    scene.spheres = AllocN(Sphere, scene.sphere_count);
    scene.planes = AllocN(Plane, scene.plane_count);
    scene.lights = AllocN(Light, scene.light_count);
    Sphere* sphere = scene.spheres;

    u8 gap = SPHERE_RADIUS * 3;
    u8 sphere_x = 0;
    u8 sphere_z = 0;

    for (u8 z = 0; z < SPHERE_VCOUNT; z++) {
        sphere_x = 0;

        for (u8 x = 0; x < SPHERE_HCOUNT; x++) {
            sphere->radius = SPHERE_RADIUS;

            sphere->world_position = Alloc(vec3);
            sphere->view_position = Alloc(vec3);

            sphere->world_position->x = sphere_x;
            sphere->world_position->y = sphere->radius;
            sphere->world_position->z = sphere_z;

            sphere_x += gap;
            sphere++;
        }

        sphere_z += gap;
    }

    Plane* last_plane = scene.planes + scene.plane_count;
    for (Plane* plane = scene.planes; plane != last_plane; plane++) {
        plane->world_position = Alloc(vec3);
        plane->view_position = Alloc(vec3);
        plane->world_normal = Alloc(vec3);
        plane->view_normal = Alloc(vec3);

        fillVec3(plane->world_position, 0);
        fillVec3(plane->world_normal, 0);
    }

    Plane *bottom_plane = scene.planes;
    Plane *top_plane = scene.planes + 1;
    Plane *left_plane = scene.planes + 2;
    Plane *right_plane = scene.planes + 3;
    Plane *back_plane = scene.planes + 4;
    Plane *front_plane = scene.planes + 5;

    top_plane->world_position->y   = 20;
    back_plane->world_position->z  = 15;
    front_plane->world_position->z = -15;
    left_plane->world_position->x  = -15;
    right_plane->world_position->x = 15;

    bottom_plane->world_normal->y = 1;
    top_plane->world_normal->y    = -1;
    front_plane->world_normal->z  = 1;
    back_plane->world_normal->z   = -1;
    left_plane->world_normal->x   = 1;
    right_plane->world_normal->x  = -1;

    Light *key_light = scene.lights;
    Light *rim_light = scene.lights + 1;
    Light *fill_light = scene.lights + 2;

    key_light->world_position = Alloc(vec3);
    fill_light->world_position = Alloc(vec3);
    rim_light->world_position = Alloc(vec3);

    key_light->view_position = Alloc(vec3);
    rim_light->view_position = Alloc(vec3);
    fill_light->view_position = Alloc(vec3);

    key_light->world_position->x = 10;
    key_light->world_position->y = 10;
    key_light->world_position->z = -5;
    rim_light->world_position->x = 0;
    rim_light->world_position->y = 3;
    rim_light->world_position->z = 10;
    fill_light->world_position->x = -10;
    fill_light->world_position->y = 10;
    fill_light->world_position->z = -5;

    key_light->color.R = 255;
    key_light->color.G = 255;
    key_light->color.B = 200;
    rim_light->color.R = 255;
    rim_light->color.G = 128;
    rim_light->color.B = 128;
    fill_light->color.R = 200;
    fill_light->color.G = 200;
    fill_light->color.B = 255;

    key_light->intensity = 15;
    rim_light->intensity = 13;
    fill_light->intensity = 11;
}