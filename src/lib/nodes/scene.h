#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"

#define SPHERE_RADIUS 1
#define SPHERE_COUNT 4
#define LIGHT_COUNT 3
#define PLANE_COUNT 6
#define LAMBERT 0
#define PHONG 1
#define BLINN 2
#define REFLECTION 3
#define REFRACTION 4

Scene scene;

void initScene() {
    scene.camera = createCamera();
    scene.active_sphere_count = 0;
    scene.plane_count = PLANE_COUNT;
    scene.light_count = LIGHT_COUNT;
    scene.sphere_count = SPHERE_COUNT;
    scene.spheres = AllocN(Sphere, scene.sphere_count);
    scene.planes = AllocN(Plane, scene.plane_count);
    scene.lights = AllocN(Light, scene.light_count);
    Sphere* sphere;

    u8 material_id = 0;
    f32 radius = 1;

    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (sphere = scene.spheres; sphere != last_sphere; sphere++) {
        sphere->radius = radius;
        sphere->material_id = material_id;
        sphere->position = Alloc(vec3);
        sphere->position->y = radius;
        radius++;
        material_id++;
    }
    scene.spheres[3].material_id = REFRACTION;

    // Back-left sphere position:
    sphere = scene.spheres;
    vec3 *pos = sphere->position;
    pos->x = -1;
    pos->z = 5;

    // Back-right sphere position:
    sphere++;
    pos = sphere->position;
    pos->x = 3;
    pos->z = 4;

    // Front-left sphere position:
    sphere++;
    pos = sphere->position;
    pos->x = -3;
    pos->z = 0;

    // Front-right sphere position:
    sphere++;
    pos = sphere->position;
    pos->x = 4;
    pos->z = -3;

    Plane* last_plane = scene.planes + scene.plane_count;
    for (Plane* plane = scene.planes; plane != last_plane; plane++) {
        plane->position = Alloc(vec3);
        plane->normal = Alloc(vec3);
        plane->material_id = LAMBERT;
        fillVec3(plane->position, 0);
        fillVec3(plane->normal, 0);
    }

    Plane *bottom_plane = scene.planes;
    Plane *top_plane = scene.planes + 1;
    Plane *left_plane = scene.planes + 2;
    Plane *right_plane = scene.planes + 3;
    Plane *back_plane = scene.planes + 4;
    Plane *front_plane = scene.planes + 5;

    top_plane->position->y   = 20;
    back_plane->position->z  = 15;
    front_plane->position->z = -15;
    left_plane->position->x  = -15;
    right_plane->position->x = 15;

    bottom_plane->normal->y = 1;
    top_plane->normal->y    = -1;
    front_plane->normal->z  = 1;
    back_plane->normal->z   = -1;
    left_plane->normal->x   = 1;
    right_plane->normal->x  = -1;

    Light *key_light = scene.lights;
    Light *rim_light = scene.lights + 1;
    Light *fill_light = scene.lights + 2;

    key_light->position = Alloc(vec3);
    fill_light->position = Alloc(vec3);
    rim_light->position = Alloc(vec3);

    key_light->position->x = 10;
    key_light->position->y = 10;
    key_light->position->z = -5;
    rim_light->position->x = 0;
    rim_light->position->y = 3;
    rim_light->position->z = 10;
    fill_light->position->x = -10;
    fill_light->position->y = 10;
    fill_light->position->z = -5;

    key_light->color.x = 255;
    key_light->color.y = 255;
    key_light->color.z = 200;
    rim_light->color.x = 255;
    rim_light->color.y = 128;
    rim_light->color.z = 128;
    fill_light->color.x = 200;
    fill_light->color.y = 200;
    fill_light->color.z = 255;

    key_light->intensity = 15;
    rim_light->intensity = 13;
    fill_light->intensity = 11;
}