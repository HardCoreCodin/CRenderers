#pragma once

#include "lib/core/types.h"

#define GEO_TYPE_COUNT 3
#define TETRAHEDRON_COUNT 4
#define CUBE_COUNT 4
#define SPHERE_COUNT 4
#define MAX_GEO_COUNT 4

#define POINT_LIGHT_COUNT 3
#define PLANE_COUNT 6
#define MATERIAL_COUNT 7

#define LAMBERT 1
#define PHONG 2
#define BLINN 4
#define REFLECTION 8
#define REFRACTION 16
#define TRANSPARENCY 32

#define IOR_AIR 1
#define IOR_GLASS 1.5f

#define GeoTypeCube 0
#define GeoTypeSphere 1
#define GeoTypeTetrahedron 2

typedef struct {
    u8 id, type, material_id;
} Geometry;

// Primitives:
// ==========

typedef struct {
    Geometry geo;
    vec3 position;
    f32 radius;
} Node;

typedef struct {
    Node node;
    vec3 normal;
} Plane;

typedef struct {
    Node node;
    mat3 rotation;
} Sphere;

typedef struct {
    Node node;
    vec3 vertices[8];
    mat3 tangent_to_world[6],
         world_to_tangent[6];
} Cube;

typedef struct {
    Node node;
    vec3 vertices[4];
    mat3 tangent_to_world[4],
         world_to_tangent[4];
} Tetrahedron;



// Materials:
// =========
typedef struct {
    bool diffuse,
         specular,
         reflection,
         refraction;
} MaterialHas;

typedef struct {
    bool blinn,
         phong;
} MaterialUses;

typedef struct {
    MaterialHas has;
    MaterialUses uses;
} MaterialSpec;
#define decodeMaterialSpec(mat, uses) \
        mat.uses.phong = uses & (u8)PHONG; \
        mat.uses.blinn = uses & (u8)BLINN; \
        mat.has.diffuse = uses & (u8)LAMBERT; \
        mat.has.specular = mat.uses.phong || mat.uses.blinn; \
        mat.has.reflection = uses & (u8)REFLECTION; \
        mat.has.refraction = uses & (u8)REFRACTION

typedef struct {
    vec3 diffuse_color;
    f32 diffuse_intensity,
        specular_intensity;
    u8 specular_exponent,
       uses;
    f32 n1_over_n2,
        n2_over_n1;
} Material;
#define decodeMaterial(material, mat, di, si, exp) \
        u8 uses = material->uses; \
        decodeMaterialSpec(mat, uses); \
        di = material->diffuse_intensity; \
        si = material->specular_intensity; \
        exp = material->specular_exponent

// Lights:
// ======
typedef struct {
    vec3 color;
} AmbientLight;
typedef struct {
    vec3 color;
    vec3 position;
    f32 intensity;
} PointLight;

// Indices:
// ========
typedef struct {
    u8 v1, v2, v3, v4;
} Indices;

typedef Node* NodePtr;
typedef struct {
    NodePtr cubes[CUBE_COUNT];
    NodePtr spheres[SPHERE_COUNT];
    NodePtr tetrahedra[TETRAHEDRON_COUNT];
} NodePointers;

vec3 tetrahedron_initial_vertex_positions[4] = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
};
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

Indices cube_indices[6];
Indices tetrahedron_indices[4];

// Scene:
// =====
typedef struct {
    AmbientLight *ambient_light;
    PointLight *point_lights;
    Tetrahedron *tetrahedra;
    Material *materials;
    Sphere *spheres;
    Plane *planes;
    Cube *cubes;
    Indices *cube_indices;
    Indices *tetrahedron_indices;
    NodePointers node_ptrs;
} Scene;

Scene main_scene;

#ifdef __CUDACC__
    __constant__ PointLight d_point_lights[POINT_LIGHT_COUNT];
    __constant__ Material d_materials[MATERIAL_COUNT];
    __constant__ Sphere d_spheres[SPHERE_COUNT];
    __constant__ Plane d_planes[PLANE_COUNT];
    __constant__ Cube d_cubes[CUBE_COUNT];
    __constant__ Tetrahedron d_tetrahedra[TETRAHEDRON_COUNT];
    __constant__ AmbientLight d_ambient_light[1];
    __constant__ Indices d_tetrahedron_indices[4];
    __constant__ Indices d_cube_indices[6];
#endif