#pragma once

#include "lib/core/types.h"
#include "lib/globals/scene.h"
#include "lib/memory/allocators.h"
#include "lib/math/math3D.h"

#include "node.h"
#include "camera.h"

void initGeometryMetadata() {
    tetrahedron_initial_vertex_positions[0].x = 0;
    tetrahedron_initial_vertex_positions[0].y = 0;
    tetrahedron_initial_vertex_positions[0].z = 0;

    tetrahedron_initial_vertex_positions[1].x = 0;
    tetrahedron_initial_vertex_positions[1].y = 1;
    tetrahedron_initial_vertex_positions[1].z = 1;

    tetrahedron_initial_vertex_positions[2].x = 1;
    tetrahedron_initial_vertex_positions[2].y = 1;
    tetrahedron_initial_vertex_positions[2].z = 0;

    tetrahedron_initial_vertex_positions[3].x = 1;
    tetrahedron_initial_vertex_positions[3].y = 0;
    tetrahedron_initial_vertex_positions[3].z = 1;

    // Front:
    // ======
    // Bottom Left
    cube_initial_vertex_positions[0].x = 0;
    cube_initial_vertex_positions[0].y = 0;
    cube_initial_vertex_positions[0].z = 0;

    // Top Left
    cube_initial_vertex_positions[1].x = 0;
    cube_initial_vertex_positions[1].y = 1;
    cube_initial_vertex_positions[1].z = 0;

    // Top Right
    cube_initial_vertex_positions[2].x = 1;
    cube_initial_vertex_positions[2].y = 1;
    cube_initial_vertex_positions[2].z = 0;

    // Bottom Right
    cube_initial_vertex_positions[3].x = 1;
    cube_initial_vertex_positions[3].y = 0;
    cube_initial_vertex_positions[3].z = 0;

    // Back
    // Bottom Left
    cube_initial_vertex_positions[4].x = 0;
    cube_initial_vertex_positions[4].y = 0;
    cube_initial_vertex_positions[4].z = 1;

    // Top Left
    cube_initial_vertex_positions[5].x = 0;
    cube_initial_vertex_positions[5].y = 1;
    cube_initial_vertex_positions[5].z = 1;

    // Top Right
    cube_initial_vertex_positions[6].x = 1;
    cube_initial_vertex_positions[6].y = 1;
    cube_initial_vertex_positions[6].z = 1;

    // Bottom Right
    cube_initial_vertex_positions[7].x = 1;
    cube_initial_vertex_positions[7].y = 0;
    cube_initial_vertex_positions[7].z = 1;

    cube_indices[0].v1 = 0;
    cube_indices[0].v2 = 1;
    cube_indices[0].v3 = 2;
    cube_indices[0].v4 = 3;

    cube_indices[1].v1 = 3;
    cube_indices[1].v2 = 2;
    cube_indices[1].v3 = 6;
    cube_indices[1].v4 = 7;

    cube_indices[2].v1 = 7;
    cube_indices[2].v2 = 6;
    cube_indices[2].v3 = 5;
    cube_indices[2].v4 = 4;

    cube_indices[3].v1 = 4;
    cube_indices[3].v2 = 5;
    cube_indices[3].v3 = 1;
    cube_indices[3].v4 = 0;

    cube_indices[4].v1 = 1;
    cube_indices[4].v2 = 5;
    cube_indices[4].v3 = 6;
    cube_indices[4].v4 = 2;

    cube_indices[5].v1 = 4;
    cube_indices[5].v2 = 0;
    cube_indices[5].v3 = 3;
    cube_indices[5].v4 = 7;

    tetrahedron_indices[0].v1 = 0;
    tetrahedron_indices[0].v2 = 1;
    tetrahedron_indices[0].v3 = 2;

    tetrahedron_indices[1].v1 = 0;
    tetrahedron_indices[1].v2 = 2;
    tetrahedron_indices[1].v3 = 3;

    tetrahedron_indices[2].v1 = 0;
    tetrahedron_indices[2].v2 = 3;
    tetrahedron_indices[2].v3 = 1;

    tetrahedron_indices[3].v1 = 3;
    tetrahedron_indices[3].v2 = 2;
    tetrahedron_indices[3].v3 = 1;

}

void initScene(Scene *scene) {
    initGeometryMetadata();
    scene->cube_indices = cube_indices;
    scene->tetrahedron_indices = tetrahedron_indices;
    scene->tetrahedra = AllocN(Tetrahedron, TETRAHEDRON_COUNT);
    scene->point_lights = AllocN(PointLight, POINT_LIGHT_COUNT);
    scene->materials = AllocN(Material, MATERIAL_COUNT);
    scene->spheres = AllocN(Sphere, SPHERE_COUNT);
    scene->planes = AllocN(Plane, PLANE_COUNT);
    scene->cubes = AllocN(Cube, CUBE_COUNT);
    scene->ambient_light = Alloc(AmbientLight);
    scene->ambient_light->color.x = 0.008f;
    scene->ambient_light->color.y = 0.008f;
    scene->ambient_light->color.z = 0.014f;

    u8 wall_material_id = 0;
    u8 diffuse_material_id = 1;
    u8 phong_material_id = 2;
    u8 blinn_material_id = 3;
    u8 reflective_material_id = 4;
    u8 refractive_material_id = 5;
    u8 reflective_refractive_material_id = 6;

    Material *walls_material = scene->materials + wall_material_id,
            *diffuse_material = scene->materials + diffuse_material_id,
            *phong_material = scene->materials + phong_material_id,
            *blinn_material = scene->materials + blinn_material_id,
            *reflective_material = scene->materials + reflective_material_id,
            *refractive_material = scene->materials + refractive_material_id,
            *reflective_refractive_material = scene->materials + reflective_refractive_material_id;

    walls_material->uses = LAMBERT;
    diffuse_material->uses = LAMBERT;
    phong_material->uses = LAMBERT | PHONG | TRANSPARENCY;
    blinn_material->uses = LAMBERT | BLINN;
    reflective_material->uses = LAMBERT | BLINN | REFLECTION;
    refractive_material->uses = LAMBERT | BLINN | REFRACTION;
    reflective_refractive_material->uses = BLINN | REFLECTION | REFRACTION;

    Material* material = scene->materials;
    for (int i = 0; i < MATERIAL_COUNT; i++, material++) {
        fillVec3(&material->diffuse_color, 1);
        material->diffuse_intensity = 1;
        material->specular_intensity = 1;
        material->specular_exponent = material->uses & (u8)BLINN ? 16 : 4;
        material->n1_over_n2 = IOR_AIR / IOR_GLASS;
        material->n2_over_n1 = IOR_GLASS / IOR_AIR;
    }

    phong_material->diffuse_color.z = 0.4f;
    diffuse_material->diffuse_color.x = 0.3f;
    diffuse_material->diffuse_color.z = 0.2f;
    diffuse_material->diffuse_color.z = 0.5f;

    Node *node;
    vec3 positions[4];
    vec3 *position;

    xform3 xf;
    mat3 *rotation = &xf.rotation_matrix;

    // Back-right cube position:
    Cube *cube = scene->cubes;
    position = positions;
    node = &cube->node;
    node->geo.material_id = reflective_material_id;
    position->x = 9;
    position->y = 4;
    position->z = 3;

    // Back-left cube position:
    cube++; position++;
    node = &cube->node;
    node->geo.material_id = phong_material_id;
    position->x = 10;
    position->z = 1;

    // Front-left cube position:
    cube++; position++;
    node = &cube->node;
    node->geo.material_id = reflective_material_id;
    position->x = -1;
    position->z = -5;

    // Front-right cube position:
    cube++; position++;
    node = &cube->node;
    node->geo.material_id = blinn_material_id;
    position->x = 10;
    position->z = -8;

    initXform3(&xf);
    position = positions;
    for (u8 i = 0; i < CUBE_COUNT; i++, position++) {
        node = scene->node_ptrs.cubes[i] = &scene->cubes[i].node;
        node->geo.id = i;
        node->geo.type = GeoTypeCube;

        fillVec3(&node->position, 0);
        initNode(node, (f32)(i + 1));

        rotateXform3(&xf, 0.3f, 0.4f, 0.5f);
        rotateNode(node, rotation);

        if (i) position->y = node->radius;
        setNodePosition(node, position);
    }


    // Back-right tetrahedron position:
    Tetrahedron *tet = scene->tetrahedra;
    node = &tet->node;
    node->geo.material_id = reflective_material_id;
    position = positions;
    position->x = 3;
    position->y = 4;
    position->z = 8;

    // Back-left tetrahedron position:
    tet++; position++;
    node = &tet->node;
    node->geo.material_id = phong_material_id;
    position->x = 4;
    position->z = 6;

    // Front-left tetrahedron position:
    tet++; position++;
    node = &tet->node;
    node->geo.material_id = reflective_material_id;
    position->x = -3;
    position->z = 0;

    // Front-right tetrahedron position:
    tet++; position++;
    node = &tet->node;
    node->geo.material_id = blinn_material_id;
    position->x = 4;
    position->z = -3;

    initXform3(&xf);
    position = positions;
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++, position++) {
        node = scene->node_ptrs.tetrahedra[i] = &scene->tetrahedra[i].node;
        node->geo.id = i;
        node->geo.type = GeoTypeTetrahedron;

        fillVec3(&node->position, 0);
        initNode(node, (f32)(i + 1));

        rotateXform3(&xf, 0.3f, 0.4f, 0.5f);
        rotateNode(node, rotation);

        if (i) position->y = node->radius;
        setNodePosition(node, position);
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
    vec3 *pos = &sphere->node.position;
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

//    bottom_plane->node.geo.material_id = reflective_material_id;

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
    key_light->color.z = 0.65f;
    rim_light->color.x = 1;
    rim_light->color.y = 0.25f;
    rim_light->color.z = 0.25f;
    fill_light->color.x = 0.65f;
    fill_light->color.y = 0.65f;
    fill_light->color.z = 1;

    key_light->intensity = 1.3f * 3;
    rim_light->intensity = 1.5f * 3;
    fill_light->intensity = 1.1f * 3;

#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_cube_indices, scene->cube_indices, sizeof(Indices) * 6, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_tetrahedron_indices, scene->tetrahedron_indices, sizeof(Indices) * 4, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_ambient_light, scene->ambient_light, sizeof(AmbientLight), 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_point_lights, scene->point_lights, sizeof(PointLight) * POINT_LIGHT_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_tetrahedra, scene->tetrahedra, sizeof(Tetrahedron) * TETRAHEDRON_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_materials, scene->materials, sizeof(Material) * MATERIAL_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_spheres, scene->spheres, sizeof(Sphere) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_planes, scene->planes, sizeof(Plane) * PLANE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_cubes, scene->cubes, sizeof(Cube) * CUBE_COUNT, 0, cudaMemcpyHostToDevice));
#endif
}