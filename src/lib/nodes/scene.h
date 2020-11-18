#pragma once

#include "lib/core/types.h"
#include "lib/memory/allocators.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"
#include "cube.h"
#include "tetrahedron.h"

#define TETRAHEDRON_COUNT 1
#define CUBE_COUNT 1
#define SPHERE_COUNT 4
#define VERTEX_COUNT 8
#define TRIANGLE_COUNT 12
#define LIGHT_COUNT 3
#define PLANE_COUNT 6
#define MATERIAL_COUNT 7
#define MAX_DISTANCE 10000

Scene scene;

void initScene() {
    scene.camera = createCamera();
    scene.active_sphere_count = 0;
    scene.plane_count = PLANE_COUNT;
    scene.light_count = LIGHT_COUNT;
    scene.sphere_count = SPHERE_COUNT;
    scene.cube_count = CUBE_COUNT;
    scene.tetrahedron_count = TETRAHEDRON_COUNT;
    scene.tetrahedra = AllocN(Tetrahedron, scene.tetrahedron_count);
    scene.spheres = AllocN(Sphere, scene.sphere_count);
    scene.planes = AllocN(Plane, scene.plane_count);
    scene.cubes = AllocN(Cube, scene.cube_count);

    scene.vertex_count = VERTEX_COUNT;
    scene.triangle_count = TRIANGLE_COUNT;
    scene.vertices = AllocN(vec3, scene.vertex_count);
    scene.triangles = AllocN(Triangle, scene.triangle_count);

    scene.point_lights = AllocN(PointLight, scene.light_count);
    scene.materials = AllocN(Material, MATERIAL_COUNT);

    initCubeData(scene.vertices, scene.triangles);

    Cube* cube = scene.cubes;
    initCube(cube);
    cube = scene.cubes;
    cube->material = scene.materials;
    cube->position.x = 0;
    cube->position.y = 5;
    cube->position.z = 0;
    for (u8 i = 0; i < 8; i++) iaddVec3(&cube->vertices[i], &cube->position);

    Tetrahedron* tetrahedron = scene.tetrahedra;
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tetrahedron++) initTetrahedron(tetrahedron);

    scene.tetrahedra->material = scene.materials;
    scene.tetrahedra->position.x = 3;
    scene.tetrahedra->position.y = 4;
    scene.tetrahedra->position.z = 8.5;

    f32 radius = 1;
    Sphere* sphere;
    Material* material = scene.materials + 1;
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    for (sphere = scene.spheres; sphere != last_sphere; sphere++, radius++, material++) {
        sphere->radius = radius;
        sphere->position.y = radius;
        sphere->material = material;
        sphere->cast_shadows = true;
        setMat3ToIdentity(&sphere->rotation_matrix);
    }
    scene.spheres[2].material = &scene.materials[4];
    scene.spheres[3].material = &scene.materials[5];
    scene.spheres[3].cast_shadows = false;

    // Back-left sphere position:
    sphere = scene.spheres;
    vec3 *pos = &sphere->position;
    pos->x = -1;
    pos->z = 5;

    // Back-right sphere position:
    sphere++;
    pos = &sphere->position;
    pos->x = 4;
    pos->z = 6;

    // Front-left sphere position:
    sphere++;
    pos = &sphere->position;
    pos->x = -3;
    pos->z = 0;

    // Front-right sphere position:
    sphere++;
    pos = &sphere->position;
    pos->x = 4;
    pos->z = -3;

    Plane* last_plane = scene.planes + scene.plane_count;
    for (Plane* plane = scene.planes; plane != last_plane; plane++) {
        plane->material = scene.materials;
        fillVec3(&plane->position, 0);
        fillVec3(&plane->normal, 0);
    }

    Plane *bottom_plane = scene.planes;
    Plane *top_plane = scene.planes + 1;
    Plane *left_plane = scene.planes + 2;
    Plane *right_plane = scene.planes + 3;
    Plane *back_plane = scene.planes + 4;
    Plane *front_plane = scene.planes + 5;

    top_plane->position.y   = 20;
    back_plane->position.z  = 15;
    front_plane->position.z = -15;
    left_plane->position.x  = -15;
    right_plane->position.x = 15;

    bottom_plane->normal.y = 1;
    top_plane->normal.y    = -1;
    front_plane->normal.z  = 1;
    back_plane->normal.z   = -1;
    left_plane->normal.x   = 1;
    right_plane->normal.x  = -1;

    PointLight *key_light = scene.point_lights;
    PointLight *rim_light = scene.point_lights + 1;
    PointLight *fill_light = scene.point_lights + 2;

    key_light->position.x = 10;
    key_light->position.y = 10;
    key_light->position.z = -5;
    rim_light->position.x = 2;
    rim_light->position.y = 5;
    rim_light->position.z = 12;
    fill_light->position.x = -10;
    fill_light->position.y = 10;
    fill_light->position.z = -5;

    key_light->color.x = 255;
    key_light->color.y = 255;
    key_light->color.z = 200;
    rim_light->color.x = 255;
    rim_light->color.y = 128;
    rim_light->color.z = 128;
    fill_light->color.x = 200;
    fill_light->color.y = 200;
    fill_light->color.z = 255;

    key_light->intensity = 13;
    rim_light->intensity = 15;
    fill_light->intensity = 11;
}