#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/shapes/line.h"
#include "lib/shapes/bbox.h"
#include "lib/globals/scene.h"
#include "lib/globals/camera.h"
#include "lib/globals/raytracing.h"
#include "lib/render/shaders/intersection/AABB.h"

Pixel BLAS_line_pixel,
      TLAS_line_pixel,
      ROOT_line_pixel,
      GEO_line_pixel;

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

    BLAS_line_pixel.color.R = MAX_COLOR_VALUE;
    BLAS_line_pixel.color.G = 0;
    BLAS_line_pixel.color.B = MAX_COLOR_VALUE;
    BLAS_line_pixel.color.A = 0;

    TLAS_line_pixel.color.R = MAX_COLOR_VALUE;
    TLAS_line_pixel.color.G = MAX_COLOR_VALUE;
    TLAS_line_pixel.color.B = 0;
    TLAS_line_pixel.color.A = 0;

    ROOT_line_pixel.color.R = 0;
    ROOT_line_pixel.color.G = MAX_COLOR_VALUE;
    ROOT_line_pixel.color.B = MAX_COLOR_VALUE;
    ROOT_line_pixel.color.A = 0;

    GEO_line_pixel.color.R = MAX_COLOR_VALUE;
    GEO_line_pixel.color.G = 0;
    GEO_line_pixel.color.B = 0;
    GEO_line_pixel.color.A = 0;
}

void updateBVH(BVH *bvh, Scene *scene) {
    AABB sphere_aabbs[SPHERE_COUNT];
    AABB *aabb = sphere_aabbs;

    f32 r, x, y, z;
    Sphere *sphere = scene->spheres;
    for (u8 s = 0; s < SPHERE_COUNT; s++, sphere++, aabb++) {
        r = sphere->node.radius;
        x = sphere->node.position.x;
        y = sphere->node.position.y;
        z = sphere->node.position.z;

        aabb->min.x = x - r;
        aabb->min.y = y - r;
        aabb->min.z = z - r;

        aabb->max.x = x + r;
        aabb->max.y = y + r;
        aabb->max.z = z + r;
    }

    AABB tet_aabbs[TETRAHEDRON_COUNT];
    aabb = tet_aabbs;

    Tetrahedron *tet = scene->tetrahedra;
    for (u8 s = 0; s < TETRAHEDRON_COUNT; s++, tet++, aabb++) {
        r = tet->node.radius;
        x = tet->node.position.x;
        y = tet->node.position.y;
        z = tet->node.position.z;

        aabb->min.x = x - r;
        aabb->min.y = y - r;
        aabb->min.z = z - r;

        aabb->max.x = x + r;
        aabb->max.y = y + r;
        aabb->max.z = z + r;
    }

    bvh->nodes->geo_type = GeoTypeNone;
    bvh->nodes->geo_ids = 0;
    bvh->nodes->children = 1 | 2;

    setParentAABB(bvh->nodes[1].aabb, tet_aabbs[0], tet_aabbs[1]);
    setParentAABB(bvh->nodes[2].aabb, tet_aabbs[2], tet_aabbs[3]);
    setParentAABB(bvh->nodes[0].aabb, bvh->nodes[1].aabb, bvh->nodes[2].aabb);

    bvh->nodes[1].geo_type = GeoTypeTetrahedron;
    bvh->nodes[2].geo_type = GeoTypeTetrahedron;
    bvh->nodes[1].geo_ids = 1 | 2;
    bvh->nodes[2].geo_ids = 4 | 8;
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
void setRayMasksFromBVH(Ray *ray, BVHNode *bvh_nodes) {
    BVHNode *node = bvh_nodes;
    u8 i, child, next_children, children = node->children;
    GeometryMasks visibility;
    visibility.spheres = visibility.cubes = visibility.tetrahedra = 0;

    while (children) { // Breadth-first_traversal
        next_children = 0;
        child = 1;

        for (i = 0; i < 8; i++, child <<= (u8)1)
            if (child & children) {
                node = bvh_nodes + i + 1;
                if (hitAABB(&node->aabb, ray)) {
                    next_children |= node->children;
                    switch (node->geo_type) {
                        case GeoTypeCube       : visibility.cubes      |= node->geo_ids; break;
                        case GeoTypeSphere     : visibility.spheres    |= node->geo_ids; break;
                        case GeoTypeTetrahedron: visibility.tetrahedra |= node->geo_ids; break;
                        case GeoTypeNone       :                                         break;
                    }
                }
            }

        children = next_children;
    }

    ray->masks.visibility = visibility;
}


//#ifdef __CUDACC__
//__device__
//__host__
////__forceinline__
//#else
//inline
//#endif
//void setRayMasksFromBVH(Ray *ray, BVH *bvh) {
//    u8 masks[3];
//    masks[0] = 0;
//    masks[1] = 1;
//    masks[2] = 2;
//
//    BVHNode *node1 = &bvh->nodes[1];
//    BVHNode *node2 = &bvh->nodes[2];
//
//    u8 geo_type1 = (u8)node1->geo_type;
//    u8 geo_type2 = (u8)node2->geo_type;
//
//    if (geo_type1 >= 0 && geo_type1 < 3) masks[geo_type1] |= node1->geo_ids;
//    if (geo_type2 >= 0 && geo_type2 < 3) masks[geo_type2] |= node2->geo_ids;
//
//    ray->masks.visibility[0] = masks[0];
//    ray->masks.visibility[1] = masks[1];
//    ray->masks.visibility[2] = masks[2];
//}
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//void setRayMasksFromBVH1(Ray *ray, BVH *bvh) {
//    u8 visibility_0 = 0,
//       visibility_1 = 0,
//       visibility_2 = 0;
//
//    BVHNode *node = bvh->nodes;
//    u8 i, child, next_children, children = node->children;
//
//    while (children) { // Breadth-first_traversal
//        next_children = 0;
//        child = 1;
//
//        for (i = 0; i < 8; i++, child <<= (u8)1)
//            if (child & children) {
//                node = &bvh->nodes[i+1];
//                if (hitAABB(&node->aabb, ray)) {
//                    next_children |= node->children;
//                    switch (node->geo_type) {
//                        case GeoTypeCube       : visibility_0 |= node->geo_ids; break;
//                        case GeoTypeSphere     : visibility_1 |= node->geo_ids; break;
//                        case GeoTypeTetrahedron: visibility_2 |= node->geo_ids; break;
//                        case GeoTypeNone       :                                break;
//                    }
//                }
//            }
//
//        children = next_children;
//    }
//
//    ray->masks.visibility[0] = visibility_0;
//    ray->masks.visibility[1] = visibility_1;
//    ray->masks.visibility[2] = visibility_2;
//}
//
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//void setRayMasksFromBVHCleanest(Ray *ray, BVH *bvh) {
//    BVHNode *node = bvh->nodes;
//    u8 i, child, next_children, children = node->children;
//
//    u8 masks[GEO_TYPE_COUNT];
//    masks[0] = 0;
//    masks[1] = 1;
//    masks[2] = 2;
//
//    while (children) { // Breadth-first_traversal
//        next_children = 0;
//        child = 1;
//
//        for (i = 0; i < 8; i++, child <<= (u8)1)
//            if (child & children) {
//                node = &bvh->nodes[i+1];
//                if (hitAABB(&node->aabb, ray)) {
//                    next_children |= node->children;
//                    if (node->geo_type != GeoTypeNone) masks[node->geo_type] |= node->geo_ids;
//                }
//            }
//
//        children = next_children;
//    }
//
//    ray->masks.visibility[0] = masks[0];
//    ray->masks.visibility[1] = masks[1];
//    ray->masks.visibility[2] = masks[2];
//}

void drawBVH(BVH *bvh, Camera *camera) {
    BBox bbox;
    BVHNode *node = bvh->nodes + 1;
    for (u8 node_id = 1; node_id < bvh->node_count; node_id++, node++) {
        setBBoxFromAABB(&node->aabb, &bbox);
        projectBBox(&bbox, camera);
        drawBBox(&bbox, node->geo_type == GeoTypeNone ? &TLAS_line_pixel : &BLAS_line_pixel);
    }
    setBBoxFromAABB(&bvh->nodes->aabb, &bbox);
    projectBBox(&bbox, camera);
    drawBBox(&bbox, &ROOT_line_pixel);
}