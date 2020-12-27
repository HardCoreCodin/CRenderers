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


enum GeometryType {
    GeoTypeNone = -1,

    GeoTypeCube,
    GeoTypeSphere,
    GeoTypeTetrahedron
};

typedef struct {
    enum GeometryType type;
    u8 id, material_id, vertex_count, prim_count;
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
    mat3 tangent_to_world[4],
         world_to_tangent[4];
    vec3 vertices[4];
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
    u8 v1, v2, v3;
} TriangleIndices;

typedef struct {
    u8 v1, v2, v3, v4;
} QuadIndices;

typedef Node* NodePtr;
typedef struct {
    NodePtr cubes[CUBE_COUNT];
    NodePtr spheres[SPHERE_COUNT];
    NodePtr tetrahedra[TETRAHEDRON_COUNT];
} NodePointers;

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
    QuadIndices *cube_indices;
    TriangleIndices *tetrahedron_indices;
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
    __constant__ TriangleIndices d_tetrahedron_indices[4];
    __constant__ QuadIndices d_cube_indices[6];
#endif