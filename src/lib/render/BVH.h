#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/shapes/line.h"
#include "lib/shapes/bbox.h"
#include "lib/globals/scene.h"
#include "lib/globals/camera.h"
#include "lib/globals/raytracing.h"
#include "lib/render/shaders/intersection/AABB.h"

#define setParentAABB(parent, child_1, child_2) \
    parent.min.x = min(child_1.min.x, child_2.min.x); \
    parent.min.y = min(child_1.min.y, child_2.min.y); \
    parent.min.z = min(child_1.min.z, child_2.min.z); \
    parent.max.x = max(child_1.max.x, child_2.max.x); \
    parent.max.y = max(child_1.max.y, child_2.max.y); \
    parent.max.z = max(child_1.max.z, child_2.max.z)

void initBVH(BVH *bvh, u8 node_count) {
    bvh->node_count = node_count;
    bvh->nodes = AllocN(BVHNode, MAX_BVH_NODE_COUNT);
}

void updateBVH(BVH *bvh, Scene *scene) {
    AABB cube_aabbs[CUBE_COUNT];
    AABB sphere_aabbs[SPHERE_COUNT];
    AABB tet_aabbs[TETRAHEDRON_COUNT];

    for (u8 i = 0; i < CUBE_COUNT;        i++) setAABBfromNode(cube_aabbs   + i, &scene->cubes[i].node);
    for (u8 i = 0; i < SPHERE_COUNT;      i++) setAABBfromNode(sphere_aabbs + i, &scene->spheres[i].node);
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) setAABBfromNode(tet_aabbs    + i, &scene->tetrahedra[i].node);

    setParentAABB(bvh->nodes[1].aabb, tet_aabbs[0], tet_aabbs[1]);
    setParentAABB(bvh->nodes[2].aabb, tet_aabbs[2], tet_aabbs[3]);

    bvh->nodes[1].geo_type = GeoTypeTetrahedron;
    bvh->nodes[2].geo_type = GeoTypeTetrahedron;
    bvh->nodes[1].geo_ids = 1 | 2;
    bvh->nodes[2].geo_ids = 4 | 8;


    setParentAABB(bvh->nodes[3].aabb, cube_aabbs[0], cube_aabbs[1]);
    setParentAABB(bvh->nodes[4].aabb, cube_aabbs[2], cube_aabbs[3]);

    bvh->nodes[3].geo_type = GeoTypeCube;
    bvh->nodes[4].geo_type = GeoTypeCube;
    bvh->nodes[3].geo_ids = 1 | 2;
    bvh->nodes[4].geo_ids = 4 | 8;

    AABB root_aabb;
    setParentAABB(root_aabb, bvh->nodes[1].aabb, bvh->nodes[2].aabb);
    setParentAABB(bvh->nodes[0].aabb, bvh->nodes[3].aabb, bvh->nodes[4].aabb);
    setParentAABB(bvh->nodes[0].aabb, bvh->nodes[0].aabb, root_aabb);

    bvh->nodes->geo_type = 0;
    bvh->nodes->geo_ids = 0;
    bvh->nodes->children = 1 | 2 | 3 | 4;
#ifdef __CUDACC__
    copyBVHNodesFromCPUtoGPU(bvh->nodes);
#endif
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
GeometryMasks getRayVisibilityMasksFromBVH(vec3 *Ro, vec3 *RD_rcp, BVHNode *bvh_nodes) {
    BVHNode *node = bvh_nodes;
    u8 i, child, next_children, children = node->children;
    GeometryMasks visibility;
    visibility.spheres = visibility.cubes = visibility.tetrahedra = 0;

    while (children) { // Breadth-first_traversal
        next_children = 0;
        child = 1;

        for (i = 0; i < 8; i++, child <<= (u8)1)
            if (child & children) {
                node = &bvh_nodes[i + 1];
                if (hitAABB(&node->aabb.min, &node->aabb.max, Ro, RD_rcp)) {
                    next_children |= node->children;
                    switch (node->geo_type) {
                        case GeoTypeCube       : visibility.cubes      |= node->geo_ids; break;
                        case GeoTypeSphere     : visibility.spheres    |= node->geo_ids; break;
                        case GeoTypeTetrahedron: visibility.tetrahedra |= node->geo_ids; break;
                    }
                }
            }

        children = next_children;
    }

    return visibility;
}

void drawBVH(BVH *bvh, Camera *camera) {
    BBox bbox;
    BVHNode *node = bvh->nodes + 1;
    Pixel pixel;
    for (u8 node_id = 1; node_id < bvh->node_count; node_id++, node++) {
        setBBoxFromAABB(&node->aabb, &bbox);
        projectBBox(&bbox, camera);
        switch (node->geo_type) {
            case GeoTypeCube: pixel.color = CYAN; break;
            case GeoTypeSphere: pixel.color = YELLOW; break;
            case GeoTypeTetrahedron: pixel.color = MAGENTA; break;
            default: pixel.color = WHITE; break;
        }
        drawBBox(&bbox, &pixel);
    }
    setBBoxFromAABB(&bvh->nodes->aabb, &bbox);
    projectBBox(&bbox, camera);

    pixel.color = BLUE;
    drawBBox(&bbox, &pixel);
}