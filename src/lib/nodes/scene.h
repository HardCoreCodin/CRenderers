#pragma once

#include "lib/core/types.h"
#include "lib/globals/scene.h"
#include "lib/memory/allocators.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"

#include "cube.h"
#include "tetrahedron.h"

void initScene(Scene *scene) {
    scene->cube_indices = AllocN(QuadIndices, 6);
    scene->tetrahedron_indices = AllocN(TriangleIndices, 4);
    scene->tetrahedra = AllocN(Tetrahedron, TETRAHEDRON_COUNT);
    scene->point_lights = AllocN(PointLight, POINT_LIGHT_COUNT);
    scene->materials = AllocN(Material, MATERIAL_COUNT);
    scene->spheres = AllocN(Sphere, SPHERE_COUNT);
    scene->planes = AllocN(Plane, PLANE_COUNT);
    scene->cubes = AllocN(Cube, CUBE_COUNT);
    scene->ambient_light = Alloc(AmbientLight);
    scene->ambient_light->color.x = 0.08;
    scene->ambient_light->color.y = 0.08;
    scene->ambient_light->color.z = 0.16;

    vec3 cube_initial_vertex_positions[8] = {
            // Front
            {0, 0, 0}, // Bottom Left
            {0, 1, 0}, // Top Left
            {1, 1, 0}, // Top Right
            {1, 0, 0}, // Bottom Right

            // Back
            {0, 0, 1}, // Bottom Left
            {0, 1, 1}, // Top Left
            {1, 1, 1}, // Top Right
            {1, 0, 1}, // Bottom Right
    };
    scene->cube_indices[0].v1 = 0;
    scene->cube_indices[0].v2 = 1;
    scene->cube_indices[0].v3 = 2;
    scene->cube_indices[0].v4 = 3;

    scene->cube_indices[1].v1 = 3;
    scene->cube_indices[1].v2 = 2;
    scene->cube_indices[1].v3 = 6;
    scene->cube_indices[1].v4 = 7;

    scene->cube_indices[2].v1 = 7;
    scene->cube_indices[2].v2 = 6;
    scene->cube_indices[2].v3 = 5;
    scene->cube_indices[2].v4 = 4;

    scene->cube_indices[3].v1 = 4;
    scene->cube_indices[3].v2 = 5;
    scene->cube_indices[3].v3 = 1;
    scene->cube_indices[3].v4 = 0;

    scene->cube_indices[4].v1 = 1;
    scene->cube_indices[4].v2 = 5;
    scene->cube_indices[4].v3 = 6;
    scene->cube_indices[4].v4 = 2;

    scene->cube_indices[5].v1 = 4;
    scene->cube_indices[5].v2 = 0;
    scene->cube_indices[5].v3 = 3;
    scene->cube_indices[5].v4 = 7;

    vec3 tetrahedron_initial_vertex_positions[4] = {
            {0, 0, 0},
            {0, 1, 1},
            {1, 1, 0},
            {1, 0, 1},
    };
    scene->tetrahedron_indices[0].v1 = 0;
    scene->tetrahedron_indices[0].v2 = 1;
    scene->tetrahedron_indices[0].v3 = 2;

    scene->tetrahedron_indices[1].v1 = 0;
    scene->tetrahedron_indices[1].v2 = 2;
    scene->tetrahedron_indices[1].v3 = 3;

    scene->tetrahedron_indices[2].v1 = 0;
    scene->tetrahedron_indices[2].v2 = 3;
    scene->tetrahedron_indices[2].v3 = 1;

    scene->tetrahedron_indices[3].v1 = 3;
    scene->tetrahedron_indices[3].v2 = 2;
    scene->tetrahedron_indices[3].v3 = 1;


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

    Cube *cube = scene->cubes;
    for (u8 i = 0; i < CUBE_COUNT; i++, cube++) {
        cube->node.geo.id = i;
        cube->node.geo.type = GeoTypeCube;
        cube->node.geo.material_id = 2;
        cube->node.radius = (f32)(i + 1);
        scene->node_ptrs.cubes[i] = &cube->node;
        initCube(cube, cube->node.radius, cube_initial_vertex_positions, scene->cube_indices);
    }

    // Back-right tetrahedron position:
    cube = scene->cubes;
    vec3 *pos = &cube->node.position;
    pos->x = 3;
    pos->y = 4;
    pos->z = 8;

    // Back-left tetrahedron position:
    cube++;
    pos = &cube->node.position;
    pos->y = cube->node.radius;
    pos->x = 4;
    pos->z = 6;

    // Front-left tetrahedron position:
    cube++;
    pos = &cube->node.position;
    pos->y = cube->node.radius;
    pos->x = -3;
    pos->z = 0;

    // Front-right tetrahedron position:
    cube++;
    pos = &cube->node.position;
    pos->y = cube->node.radius;
    pos->x = 4;
    pos->z = -3;

    cube = scene->cubes;
    xform3 xf;
    initXform3(&xf);
    for (u8 i = 0; i < CUBE_COUNT; i++, cube++) {
        for (u8 v = 0; v < 8; v++) {
            imulVec3Mat3(&cube->vertices[v], &xf.rotation_matrix);
            iaddVec3(&cube->vertices[v], &cube->node.position);
        }
        for (u8 q = 0; q < 6; q++) {
            imulMat3(&cube->tangent_to_world[q], &xf.rotation_matrix);
        }

        updateCubeMatrices(cube);

        rotateXform3(&xf, 0.3f, 0.4f, 0.5f);
    }

    Tetrahedron *tet = scene->tetrahedra;
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tet++) {
        tet->node.geo.id = i;
        tet->node.geo.type = GeoTypeTetrahedron;
        tet->node.geo.material_id = 2;
        tet->node.radius = (f32)(i + 1);
        scene->node_ptrs.tetrahedra[i] = &tet->node;
        initTetrahedron(tet, tet->node.radius, tetrahedron_initial_vertex_positions, scene->tetrahedron_indices);
    }

    // Back-right tetrahedron position:
    tet = scene->tetrahedra;
    pos = &tet->node.position;
    pos->x = 3;
    pos->y = 4;
    pos->z = 8;

    // Back-left tetrahedron position:
    tet++;
    pos = &tet->node.position;
    pos->y = tet->node.radius;
    pos->x = 4;
    pos->z = 6;

    // Front-left tetrahedron position:
    tet++;
    pos = &tet->node.position;
    pos->y = tet->node.radius;
    pos->x = -3;
    pos->z = 0;

    // Front-right tetrahedron position:
    tet++;
    pos = &tet->node.position;
    pos->y = tet->node.radius;
    pos->x = 4;
    pos->z = -3;

    tet = scene->tetrahedra;
    initXform3(&xf);
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, tet++) {
        for (u8 t = 0; t < 4; t++) {
            imulVec3Mat3(&tet->vertices[t], &xf.rotation_matrix);
            iaddVec3(&tet->vertices[t], &tet->node.position);

            imulMat3(&tet->tangent_to_world[t], &xf.rotation_matrix);
        }
        updateTetrahedronMatrices(tet);

        rotateXform3(&xf, 0.3f, 0.4f, 0.5f);
    }

    Sphere* sphere = scene->spheres;
    f32 radius = 1;
    u8 material_id = 1;
    for (u8 i = 0; i < SPHERE_COUNT; i++, radius++, material_id++, sphere++) {
        sphere->node.geo.id = i;
        sphere->node.geo.type = GeoTypeTetrahedron;
        sphere->node.radius = radius;
        sphere->node.position.y = radius;
        sphere->node.geo.material_id = material_id;
        setMat3ToIdentity(&sphere->rotation);
        scene->node_ptrs.spheres[i] = &sphere->node;
    }
    scene->spheres[2].node.geo.material_id = 4;
    scene->spheres[3].node.geo.material_id = 5;

    // Back-left sphere position:
    sphere = scene->spheres;
    pos = &sphere->node.position;
    pos->x = -1;
    pos->z = 5;

    // Back-right sphere position:
    sphere++;
    pos = &sphere->node.position;
    pos->x = 4;
    pos->z = 6;

    // Front-left sphere position:
    sphere++;
    pos = &sphere->node.position;
    pos->x = -3;
    pos->z = 0;

    // Front-right sphere position:
    sphere++;
    pos = &sphere->node.position;
    pos->x = 4;
    pos->z = -3;

    Plane* plane;
    for (u8 i = 0; i < PLANE_COUNT; i++) {
        plane = &scene->planes[i];
        plane->node.geo.material_id = 0;
        fillVec3(&plane->node.position, 0);
        fillVec3(&plane->normal, 0);
    }

    Plane *bottom_plane = scene->planes;
    Plane *top_plane = scene->planes + 1;
    Plane *left_plane = scene->planes + 2;
    Plane *right_plane = scene->planes + 3;
    Plane *back_plane = scene->planes + 4;
    Plane *front_plane = scene->planes + 5;

    top_plane->node.position.y   = 20;
    back_plane->node.position.z  = 20;
    front_plane->node.position.z = -20;
    left_plane->node.position.x  = -20;
    right_plane->node.position.x = 20;

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

    key_light->color.x = 1;
    key_light->color.y = 1;
    key_light->color.z = 0.8;
    rim_light->color.x = 1;
    rim_light->color.y = 0.5;
    rim_light->color.z = 0.5;
    fill_light->color.x = 0.8;
    fill_light->color.y = 0.8;
    fill_light->color.z = 1;

    key_light->intensity = 13;
    rim_light->intensity = 15;
    fill_light->intensity = 11;

#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_cube_indices, scene->cube_indices, sizeof(QuadIndices) * 6, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_tetrahedron_indices, scene->tetrahedron_indices, sizeof(TriangleIndices) * 4, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_ambient_light, scene->ambient_light, sizeof(AmbientLight), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_point_lights, scene->point_lights, sizeof(PointLight) * POINT_LIGHT_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_tetrahedra, scene->tetrahedra, sizeof(Tetrahedron) * TETRAHEDRON_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_materials, scene->materials, sizeof(Material) * MATERIAL_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_spheres, scene->spheres, sizeof(Sphere) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_planes, scene->planes, sizeof(Plane) * PLANE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_cubes, scene->cubes, sizeof(Cube) * CUBE_COUNT, 0, cudaMemcpyHostToDevice));
#endif
}