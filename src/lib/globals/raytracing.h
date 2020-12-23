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
    vec3 *origin,
         *direction,
         *direction_rcp;
    RayHit hit;
    Masks masks;
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

typedef struct {
    u32 active_pixels;
    u8 visible_nodes[GEO_TYPE_COUNT];
} Stats;

typedef struct {
    AABB aabb;
    u8 children, geo_type, geo_ids;
} BVHNode;

typedef struct {
    u8 node_count;
    BVHNode *nodes;
} BVH;

typedef struct {
    BVH bvh;
    SSB ssb;
    Masks masks;
    Stats stats;
    u32 ray_count;
    u8 rays_per_pixel;
    vec3 *ray_directions,
         *ray_directions_rcp;
} RayTracer;
RayTracer ray_tracer;

enum RenderMode {
    Beauty,
    Normal,
    UVs
};
enum RenderMode render_mode = Beauty;

#ifdef __CUDACC__
    __device__ vec3 d_ray_directions[MAX_WIDTH * MAX_HEIGHT];
    __device__ vec3 d_ray_directions_rcp[MAX_WIDTH * MAX_HEIGHT];
    __device__ u8 d_masks_out[MAX_WIDTH * MAX_HEIGHT];

    #define copyMasksFromGPUtoCPU(masks, count) \
        gpuErrchk(cudaMemcpyFromSymbol(masks, d_masks_out, count, 0, cudaMemcpyDeviceToHost))
#endif