#pragma once

#include "lib/core/types.h"

#define GEO_TYPE_COUNT 3
#define GEO_TYPE__NONE 0
#define GEO_TYPE__SPHERE 1
#define GEO_TYPE__CUBE 2
#define GEO_TYPE__TETRAHEDRON 3

#define TETRAHEDRON_COUNT 1
#define CUBE_COUNT 1
#define SPHERE_COUNT 4
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


typedef struct {
    u8 visibility[GEO_TYPE_COUNT],
       transparency[GEO_TYPE_COUNT],
       shadowing[GEO_TYPE_COUNT];
} Masks;

// Primitives:
// ==========
typedef struct {
    vec3 position,
         normal;
    u8 material_id;
} Plane;

typedef struct {
    u8 p1,
       p2,
       p3;
    mat3 tangent_to_world,
         world_to_tangent;
} Triangle;
#define expandTriangle(t, vs, p1, p2, p3, n) p1 = &vs[t->p1]; p2 = &vs[t->p2]; p3 = &vs[t->p3]; n = &t->tangent_to_world.Z
#define expandTrianglePN(t, vs, p1, n) p1 = &vs[t->p1]; n = &t->tangent_to_world.Z

typedef struct {
    Triangle triangles[12];
    mat3 rotation_matrix;
    vec3 position, vertices[8];
    u8 material_id;
} Cube;

typedef struct {
    Triangle triangles[4];
    vec3 vertices[4];
    xform3 xform;
    u8 material_id;
} Tetrahedron;

typedef struct {
    vec3 position;
    mat3 rotation;
    f32 radius;
    u8 material_id;
} Sphere;


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
    __constant__ Masks d_masks[1];
    __constant__ mat3 d_sphere_rotations[SPHERE_COUNT];
    __constant__ vec3 d_sphere_view_positions[SPHERE_COUNT];
    __constant__ Bounds2Di d_sphere_view_bounds[SPHERE_COUNT];
//    __constant__ Scene d_scene[1];
#endif