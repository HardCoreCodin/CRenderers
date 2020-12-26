#pragma once

#include "lib/core/types.h"
#include "lib/globals/scene.h"
#include "lib/memory/allocators.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"

#include "cube.h"
#include "tetrahedron.h"

void initScene(Scene *scene) {
    scene->tetrahedra = AllocN(Tetrahedron, TETRAHEDRON_COUNT);
    scene->point_lights = AllocN(PointLight, POINT_LIGHT_COUNT);
    scene->materials = AllocN(Material, MATERIAL_COUNT);
    scene->spheres = AllocN(Sphere, SPHERE_COUNT);
    scene->planes = AllocN(Plane, PLANE_COUNT);
    scene->cubes = AllocN(Cube, CUBE_COUNT);
    scene->ambient_light = Alloc(AmbientLight);
    scene->ambient_light->color.x = 20;
    scene->ambient_light->color.y = 20;
    scene->ambient_light->color.z = 40;
    scene->index_buffers = Alloc(IndexBuffers);

    scene->index_buffers->tetrahedron[0][0] = 0;
    scene->index_buffers->tetrahedron[0][1] = 1;
    scene->index_buffers->tetrahedron[0][2] = 2;

    scene->index_buffers->tetrahedron[1][0] = 0;
    scene->index_buffers->tetrahedron[1][1] = 2;
    scene->index_buffers->tetrahedron[1][2] = 3;

    scene->index_buffers->tetrahedron[2][0] = 0;
    scene->index_buffers->tetrahedron[2][1] = 3;
    scene->index_buffers->tetrahedron[2][2] = 1;

    scene->index_buffers->tetrahedron[3][0] = 3;
    scene->index_buffers->tetrahedron[3][1] = 2;
    scene->index_buffers->tetrahedron[3][2] = 1;

    Material *walls_material = scene->materials,
            *diffuse_ball_material = scene->materials + 1,
            *specular_ball_material_phong = scene->materials + 2,
            *specular_ball_material_blinn = scene->materials + 3,
            *reflective_ball_material = scene->materials + 4,
            *refractive_ball_material = scene->materials + 5,
            *reflective_refractive_ball_material = scene->materials + 6;

    walls_material->uses = LAMBERT;
    diffuse_ball_material->uses = LAMBERT;
    specular_ball_material_phong->uses = LAMBERT | PHONG | TRANSPARENCY;
    specular_ball_material_blinn->uses = LAMBERT | BLINN;
    reflective_ball_material->uses = BLINN | REFLECTION;
    refractive_ball_material->uses = BLINN | REFRACTION;
    reflective_refractive_ball_material->uses = BLINN | REFLECTION | REFRACTION;

    Material* material = scene->materials;
    for (int i = 0; i < MATERIAL_COUNT; i++, material++) {
        fillVec3(&material->diffuse_color, 1);
        material->diffuse_intensity = 1;
        material->specular_intensity = 1;
        material->specular_exponent = material->uses & (u8)BLINN ? 16 : 4;
        material->n1_over_n2 = IOR_AIR / IOR_GLASS;
        material->n2_over_n1 = IOR_GLASS / IOR_AIR;
    }

    specular_ball_material_phong->diffuse_color.z = 0.4f;
    diffuse_ball_material->diffuse_color.x = 0.3f;
    diffuse_ball_material->diffuse_color.z = 0.2f;
    diffuse_ball_material->diffuse_color.z = 0.7f;

    for (u8 i = 0; i < CUBE_COUNT; i++) initCube(scene->cubes + i);
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) initTetrahedron(scene->tetrahedra + i, scene->index_buffers, (f32)(i + 1));

    scene->cubes->material_id = 0;
    scene->cubes->position.x = 0;
    scene->cubes->position.y = 9;
    scene->cubes->position.z = 0;
    for (u8 i = 0; i < 8; i++) iaddVec3(&scene->cubes->vertices[i], &scene->cubes->position);

    scene->tetrahedra->material_id = 2;

    f32 radius = 1;
    Sphere* sphere;
    u8 material_id = 1;
    u8 sphere_id = 1;
    for (u8 i = 0; i < SPHERE_COUNT; i++, radius++, material_id++, sphere_id <<= (u8)1) {
        sphere = &scene->spheres[i];
        sphere->radius = radius;
        sphere->position.y = radius;
        sphere->material_id = material_id;
        setMat3ToIdentity(&sphere->rotation);
    }
    scene->spheres[2].material_id = 4;
    scene->spheres[3].material_id = 5;

    // Back-right tet position:
    Tetrahedron *tet = scene->tetrahedra;
    vec3 *pos = &tet->xform.position;
    pos->y = 3;
    pos->x = 4;
    pos->z = 8;

    // Back-left tet position:
    tet++;
    pos = &tet->xform.position;
    pos->y = tet->radius;
    pos->x = -1;
    pos->z = 5;

    // Front-left tet position:
    tet++;
    pos = &tet->xform.position;
    pos->y = tet->radius;
    pos->x = -3;
    pos->z = 0;

    // Front-right tet position:
    tet++;
    pos = &tet->xform.position;
    pos->y = tet->radius;
    pos->x = 4;
    pos->z = -3;

    tet = scene->tetrahedra;
    xform3 xf;
    initXform3(&xf);
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tet++) {
        for (u8 t = 0; t < 4; t++) {
            imulVec3Mat3(&tet->vertices[t], &xf.rotation_matrix);
            iaddVec3(&tet->vertices[t], &tet->xform.position);

            imulMat3(&tet->triangles[t].tangent_to_world, &xf.rotation_matrix);
        }
        updateTetrahedronMatrices(tet);

        rotateXform3(&xf, 0.3f, 0.4f, 0.5f);
    }

    // Back-left sphere position:
    sphere = scene->spheres;
    pos = &sphere->position;
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

    Plane* plane;
    for (u8 i = 0; i < PLANE_COUNT; i++) {
        plane = &scene->planes[i];
        plane->material_id = 0;
        fillVec3(&plane->position, 0);
        fillVec3(&plane->normal, 0);
    }

    Plane *bottom_plane = scene->planes;
    Plane *top_plane = scene->planes + 1;
    Plane *left_plane = scene->planes + 2;
    Plane *right_plane = scene->planes + 3;
    Plane *back_plane = scene->planes + 4;
    Plane *front_plane = scene->planes + 5;

    top_plane->position.y   = 20;
    back_plane->position.z  = 20;
    front_plane->position.z = -20;
    left_plane->position.x  = -20;
    right_plane->position.x = 20;

    bottom_plane->normal.y = 1;
    top_plane->normal.y    = -1;
    front_plane->normal.z  = 1;
    back_plane->normal.z   = -1;
    left_plane->normal.x   = 1;
    right_plane->normal.x  = -1;

    PointLight *key_light = scene->point_lights;
    PointLight *rim_light = scene->point_lights + 1;
    PointLight *fill_light = scene->point_lights + 2;

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

#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_ambient_light, scene->ambient_light, sizeof(AmbientLight), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_point_lights, scene->point_lights, sizeof(PointLight) * POINT_LIGHT_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_tetrahedra, scene->tetrahedra, sizeof(Tetrahedron) * TETRAHEDRON_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_materials, scene->materials, sizeof(Material) * MATERIAL_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_spheres, scene->spheres, sizeof(Sphere) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_planes, scene->planes, sizeof(Plane) * PLANE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_cubes, scene->cubes, sizeof(Cube) * CUBE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_index_buffers, scene->index_buffers, sizeof(IndexBuffers), 0, cudaMemcpyHostToDevice));

//
//    Scene *d_scene_mirrored = Alloc(Scene);
//    d_scene_mirrored->ambient_light = d_ambient_light;
//    d_scene_mirrored->point_lights = d_point_lights;
//    d_scene_mirrored->tetrahedra = d_tetrahedra;
//    d_scene_mirrored->materials = d_materials;
//    d_scene_mirrored->spheres = d_spheres;
//    d_scene_mirrored->planes = d_planes;
//    d_scene_mirrored->cubes = d_cubes;
//    d_scene_mirrored->masks = d_masks;
//    cudaMemcpy(d_scene, d_scene_mirrored, sizeof(Scene), cudaMemcpyHostToDevice);
//
////    gpuErrchk(cudaMemcpyToSymbol(d_scene, d_scene_mirrored, sizeof(Scene), 0, cudaMemcpyHostToDevice));
#endif
}