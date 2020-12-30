#pragma once

#include "lib/core/types.h"
#include "lib/globals/scene.h"

#define MAX_HIT_DEPTH 4
#define MAX_DISTANCE 10000

#define FULL_MASK (1 + 2 + 4 + 8)

static char* RAY_TRACER_TITLE = "RayTrace";

typedef struct {
    vec2 uv;
    vec3 position,
         normal;
    f32 distance;
    bool is_back_facing;
    u8 material_id;
} RayHit;

typedef struct {
    u8 cubes, spheres, tetrahedra;
} GeometryMasks;

typedef struct {
    GeometryMasks visibility, transparency, shadowing;
} Masks;

typedef struct {
    Masks masks;
    RayHit hit;
    vec3 *origin,
         *direction;
} Ray;


typedef struct {
    Bounds2Di spheres[SPHERE_COUNT], cubes[CUBE_COUNT], tetrahedra[TETRAHEDRON_COUNT];
} GeometryBounds;

typedef struct {
    vec3 spheres[SPHERE_COUNT], cubes[CUBE_COUNT], tetrahedra[TETRAHEDRON_COUNT];
} GeometryViewPositions;

typedef struct {
    GeometryBounds bounds;
    GeometryViewPositions view_positions;
} SSB;

#define MAX_BVH_NODE_COUNT 8

typedef struct {
    u8 children, geo_type, geo_ids;
    AABB aabb;
} BVHNode;

typedef struct {
    u8 node_count;
    BVHNode *nodes;
} BVH;

typedef struct {
    BVH bvh;
    SSB ssb;
    Masks masks;
    u32 ray_count;
    u8 rays_per_pixel;
    vec3 *ray_directions,
         *ray_directions_rcp;
} RayTracer;
RayTracer ray_tracer;

#ifdef __CUDACC__
    __constant__ vec3 d_vectors[4];
    __constant__ Masks d_masks[1];
    __constant__ BVHNode d_bvh_nodes[MAX_BVH_NODE_COUNT];
    __constant__ GeometryBounds d_ssb_bounds[1];

    #define copyMasksFromCPUtoGPU(masks) gpuErrchk(cudaMemcpyToSymbol(d_masks, masks, sizeof(Masks), 0, cudaMemcpyHostToDevice))
    #define copyBVHNodesFromCPUtoGPU(bvh_nodes) gpuErrchk(cudaMemcpyToSymbol(d_bvh_nodes, bvh_nodes, sizeof(BVHNode) * MAX_BVH_NODE_COUNT, 0, cudaMemcpyHostToDevice))
    #define copySSBBoundsFromCPUtoGPU(ssb_bounds) gpuErrchk(cudaMemcpyToSymbol(d_ssb_bounds, ssb_bounds, sizeof(GeometryBounds), 0, cudaMemcpyHostToDevice))
#endif