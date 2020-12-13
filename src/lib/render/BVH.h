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
    bvh->nodes = AllocN(BVHNode, node_count);

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
        r = sphere->radius;
        x = sphere->position.x;
        y = sphere->position.y;
        z = sphere->position.z;

        aabb->min.x = x - r;
        aabb->min.y = y - r;
        aabb->min.z = z - r;

        aabb->max.x = x + r;
        aabb->max.y = y + r;
        aabb->max.z = z + r;
    }

    if (bvh->node_count == 1) {
        bvh->nodes->geo_type = GEO_TYPE__SPHERE;
        bvh->nodes->geo_ids = 1 | 2 | 4 | 8;
        bvh->nodes->children = 0;

        setParentAABB(sphere_aabbs[0], sphere_aabbs[0], sphere_aabbs[1]);
        setParentAABB(sphere_aabbs[1], sphere_aabbs[2], sphere_aabbs[3]);
        setParentAABB(sphere_aabbs[0], sphere_aabbs[0], sphere_aabbs[1]);

        bvh->nodes->aabb = sphere_aabbs[0];
    } else if (bvh->node_count == 3) {
        bvh->nodes->geo_type = GEO_TYPE__NONE;
        bvh->nodes->geo_ids = 0;
        bvh->nodes->children = 1 | 2;

        setParentAABB(bvh->nodes[1].aabb, sphere_aabbs[0], sphere_aabbs[1]);
        setParentAABB(bvh->nodes[2].aabb, sphere_aabbs[2], sphere_aabbs[3]);
        setParentAABB(bvh->nodes[0].aabb, bvh->nodes[1].aabb, bvh->nodes[2].aabb);

        bvh->nodes[1].geo_type = GEO_TYPE__SPHERE;
        bvh->nodes[2].geo_type = GEO_TYPE__SPHERE;
        bvh->nodes[1].geo_ids = 1 | 2;
        bvh->nodes[2].geo_ids = 4 | 8;
    }
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void setRayMasksFromBVH(Ray *ray, BVH *bvh) {
    for (u8 i = 0; i < GEO_TYPE_COUNT; i++) ray->masks.visibility[i] = 0;
    if (bvh->nodes->geo_type) ray->masks.visibility[bvh->nodes->geo_type - 1] |= bvh->nodes->geo_ids;
    if (bvh->nodes->children) {
        BVHNode *node = bvh->nodes;
        u8 i, child, next_children, children = node->children;

        while (children) { // Breadth-first_traversal
            next_children = 0;
            child = 1;

            for (i = 0; i < 8; i++, child <<= (u8)1)
                if (child & children) {
                    node = &bvh->nodes[i+1];
                    if (hitAABB(&node->aabb, ray)) {
                        next_children |= node->children;
                        if (node->geo_type) ray->masks.visibility[node->geo_type - 1] |= node->geo_ids;
                    }
                }

            children = next_children;
        }
    }
}

void drawBVH(BVH *bvh, Camera *camera) {
    BBox bbox;
    if (bvh->node_count > 1) {
        BVHNode *node = bvh->nodes + 1;
        for (u8 node_id = 1; node_id < bvh->node_count; node_id++, node++) {
            if (node->geo_type) {
                setBBoxFromAABB(&node->aabb, &bbox);
                projectBBox(&bbox, camera);
                drawBBox(&bbox, &BLAS_line_pixel);
            }
        }
        node = bvh->nodes + 1;
        for (u8 node_id = 1; node_id < bvh->node_count; node_id++, node++) {
            if (!node->geo_type) {
                setBBoxFromAABB(&node->aabb, &bbox);
                projectBBox(&bbox, camera);
                drawBBox(&bbox, &TLAS_line_pixel);
            }
        }
    }
    setBBoxFromAABB(&bvh->nodes->aabb, &bbox);
    projectBBox(&bbox, camera);
    drawBBox(&bbox, &ROOT_line_pixel);
}


//
//typedef struct {
//    AABB aabb;
//    u8 children_bits,
//       geo_type;
//    u8 geo_ids[2];
//    u8 children[2];
//} BVHNode;
//
//typedef struct {
//    u8 depth, node_count;
//    BVHNode *nodes;
//} BVH;
//
//typedef struct {
//    u8 children;
//    bool is_leaf;
//} TLAS_Node;
//
//typedef struct {
//    u8 geo_type,
//       geo_ids;
//} BLAS_Node;
//
//typedef struct {
//    u8 TLAS_node_count,
//       BLAS_node_count, depth;
//    AABB root_aabb,
//         *TLAS_node_aabbs,
//         *BLAS_node_aabbs;
//    BLAS_Node *BLAS_nodes;
//    TLAS_Node *TLAS_nodes;
//    TLAS_Node root;
//} BVH_SoA;



//
//
//void initBVH_SoA(BVH_SoA *bvh, u8 TLAS_node_count, u8 BLAS_node_count) {
//    bvh->BLAS_node_count = BLAS_node_count;
//    bvh->TLAS_node_count = TLAS_node_count;
//    if (BLAS_node_count) bvh->BLAS_nodes = AllocN(BLAS_Node, BLAS_node_count);
//    if (TLAS_node_count) bvh->TLAS_nodes = AllocN(TLAS_Node, TLAS_node_count);
//
//    bvh->TLAS_node_aabbs = AllocN(AABB, TLAS_node_count);
//    bvh->BLAS_node_aabbs = AllocN(AABB, BLAS_node_count);
//
////    BLAS_Node *blas_node = bvh->BLAS_nodes;
////    blas_node->geo_type = GEO_TYPE__SPHERE;
////    blas_node->geo_ids = 1;
////
////    blas_node++;
////    blas_node->geo_type = GEO_TYPE__SPHERE;
////    blas_node->geo_ids = 2;
////
////    blas_node++;
////    blas_node->geo_type = GEO_TYPE__SPHERE;
////    blas_node->geo_ids = 4;
////
////    blas_node++;
////    blas_node->geo_type = GEO_TYPE__SPHERE;
////    blas_node->geo_ids = 8;
////
////
////    TLAS_Node *tlas_node = bvh->TLAS_nodes;
////    tlas_node->children = 1 | 2;
////    tlas_node->is_leaf = true;
////
////    tlas_node++;
////    tlas_node->children = 4 | 8;
////    tlas_node->is_leaf = true;
////
////
////    bvh->root.is_leaf = false;
////    bvh->root.children = 1 | 2;
//
////
////    bvh->BLAS_nodes->geo_type = GEO_TYPE__SPHERE;
////    bvh->BLAS_nodes->geo_ids = 1 | 2 | 4 | 8;
////    bvh->root.children = 1;
////    bvh->root.is_leaf = true;
//
//
//
//    BLAS_Node *blas_node = bvh->BLAS_nodes;
//    blas_node->geo_type = GEO_TYPE__SPHERE;
//    blas_node->geo_ids = 1 | 2;
//
//    blas_node++;
//    blas_node->geo_type = GEO_TYPE__SPHERE;
//    blas_node->geo_ids = 4 | 8;
//
//    bvh->root.children = 1 | 2;
//    bvh->root.is_leaf = true;
//
//
//    BLAS_line_pixel.color.R = MAX_COLOR_VALUE;
//    BLAS_line_pixel.color.G = 0;
//    BLAS_line_pixel.color.B = MAX_COLOR_VALUE;
//    BLAS_line_pixel.color.A = 0;
//
//    TLAS_line_pixel.color.R = MAX_COLOR_VALUE;
//    TLAS_line_pixel.color.G = MAX_COLOR_VALUE;
//    TLAS_line_pixel.color.B = 0;
//    TLAS_line_pixel.color.A = 0;
//
//    ROOT_line_pixel.color.R = 0;
//    ROOT_line_pixel.color.G = MAX_COLOR_VALUE;
//    ROOT_line_pixel.color.B = MAX_COLOR_VALUE;
//    ROOT_line_pixel.color.A = 0;
//
//    GEO_line_pixel.color.R = MAX_COLOR_VALUE;
//    GEO_line_pixel.color.G = 0;
//    GEO_line_pixel.color.B = 0;
//    GEO_line_pixel.color.A = 0;
//}
//
//#define setParentAABB(parent, child_1, child_2) \
//    (parent)->min.x = min((child_1)->min.x, (child_2)->min.x); \
//    (parent)->min.y = min((child_1)->min.y, (child_2)->min.y); \
//    (parent)->min.z = min((child_1)->min.z, (child_2)->min.z); \
//    (parent)->max.x = max((child_1)->max.x, (child_2)->max.x); \
//    (parent)->max.y = max((child_1)->max.y, (child_2)->max.y); \
//    (parent)->max.z = max((child_1)->max.z, (child_2)->max.z)
//
//void updateBVH_SoA(BVH_SoA *bvh, Scene *scene) {
//    AABB *aabb2, *aabb3, *aabb = bvh->BLAS_node_aabbs;
//
//    f32 r, x, y, z;
//    Sphere *sphere = scene->spheres;
//    for (u8 s = 0; s < SPHERE_COUNT; s++, sphere++, aabb++) {
//        r = sphere->radius;
//        x = sphere->position.x;
//        y = sphere->position.y;
//        z = sphere->position.z;
//
//        aabb->min.x = x - r;
//        aabb->min.y = y - r;
//        aabb->min.z = z - r;
//
//        aabb->max.x = x + r;
//        aabb->max.y = y + r;
//        aabb->max.z = z + r;
//    }
//
//    aabb = bvh->TLAS_node_aabbs;
//    aabb2 = bvh->BLAS_node_aabbs;
//    aabb3 = bvh->BLAS_node_aabbs + 1;
//    setParentAABB(aabb, aabb2, aabb3);
//
//    aabb++;
//    aabb2 += 2;
//    aabb3 += 2;
//    setParentAABB(aabb, aabb2, aabb3);
//
//    aabb = &bvh->root_aabb;
//    aabb2 = bvh->TLAS_node_aabbs;
//    aabb3 = bvh->TLAS_node_aabbs + 1;
//    setParentAABB(aabb, aabb2, aabb3);
//
//    *bvh->BLAS_node_aabbs = *bvh->TLAS_node_aabbs;
//    *(bvh->BLAS_node_aabbs + 1) = *(bvh->TLAS_node_aabbs + 1);
//
//    bvh->TLAS_node_count = 0;
//    bvh->BLAS_node_count = 2;
//
////
////    *bvh->BLAS_node_aabbs = bvh->root_aabb;
////
////    bvh->TLAS_node_count = 0;
////    bvh->BLAS_node_count = 1;
//}
//
//void updateBVH(BVH *bvh, Scene *scene) {
//    f32 r, x, y, z;
//    AABB *aabb;
//    Sphere *sphere = scene->spheres;
//    BVHNode *node = bvh->nodes + 1;
//    u8 sphere_id = 1;
//    for (u8 s = 0; s < SPHERE_COUNT; s++, sphere++, node++, sphere_id <<= (u8)1) {
//        node->geo_ids[0] = sphere_id;
//        node->geo_type = GEO_TYPE__SPHERE;
//        node->children_bits = 0;
//
//        r = sphere->radius;
//        x = sphere->position.x;
//        y = sphere->position.y;
//        z = sphere->position.z;
//
//        aabb = &node->aabb;
//        aabb->min.x = x - r;
//        aabb->min.y = y - r;
//        aabb->min.z = z - r;
//
//        aabb->max.x = x + r;
//        aabb->max.y = y + r;
//        aabb->max.z = z + r;
//    }
//
//    node = bvh->nodes + 5;
//    node->geo_ids[0] = 1;
//    node->geo_ids[1] = 2;
//    node->geo_type = GEO_TYPE__SPHERE;
//    setParentAABB(&node->aabb,
//                  &bvh->nodes[1].aabb,
//                  &bvh->nodes[2].aabb);
//
//    node = bvh->nodes + 6;
//    node->geo_ids[0] = 4;
//    node->geo_ids[1] = 8;
//    node->geo_type = GEO_TYPE__SPHERE;
//    setParentAABB(&node->aabb,
//                  &bvh->nodes[3].aabb,
//                  &bvh->nodes[4].aabb);
//
//    bvh->nodes[1] = bvh->nodes[5];
//    bvh->nodes[2] = bvh->nodes[6];
//
//    node = bvh->nodes;
//    node->children[0] = 1;
//    node->children[1] = 2;
//    node->children_bits = 1 | 2;
//    node->geo_type = GEO_TYPE__NONE;
//    setParentAABB(&node->aabb,
//                  &bvh->nodes[1].aabb,
//                  &bvh->nodes[2].aabb);
//}
//
//
//void updateBVH2(BVH *bvh, Scene *scene) {
//    f32 r, x, y, z;
//    AABB *aabb;
//    Sphere *sphere = scene->spheres;
//    BVHNode *node = bvh->nodes + 1;
//    u8 sphere_id = 1;
//    for (u8 s = 0; s < SPHERE_COUNT; s++, sphere++, node++, sphere_id <<= (u8)1) {
//        node->geo_ids[0] = sphere_id;
//        node->geo_type = GEO_TYPE__SPHERE;
//        node->children_bits = 0;
//
//        r = sphere->radius;
//        x = sphere->position.x;
//        y = sphere->position.y;
//        z = sphere->position.z;
//
//        aabb = &node->aabb;
//        aabb->min.x = x - r;
//        aabb->min.y = y - r;
//        aabb->min.z = z - r;
//
//        aabb->max.x = x + r;
//        aabb->max.y = y + r;
//        aabb->max.z = z + r;
//    }
//
//    node = bvh->nodes + 5;
//    node->children_bits = 1 | 2;
//    node->children[0] = 1;
//    node->children[1] = 2;
//    node->geo_type = GEO_TYPE__NONE;
//    setParentAABB(&node->aabb,
//                  &bvh->nodes[1].aabb,
//                  &bvh->nodes[2].aabb);
//
//    node = bvh->nodes + 6;
//    node->children[0] = 3;
//    node->children[1] = 4;
//    node->children_bits = 4 | 8;
//    node->geo_type = GEO_TYPE__NONE;
//    setParentAABB(&node->aabb,
//                  &bvh->nodes[3].aabb,
//                  &bvh->nodes[4].aabb);
//
//
//
//    node = bvh->nodes;
//    node->children[0] = 5;
//    node->children[1] = 6;
//    node->children_bits = 16 | 32;
//    node->geo_type = GEO_TYPE__NONE;
//    setParentAABB(&node->aabb,
//                  &bvh->nodes[5].aabb,
//                  &bvh->nodes[6].aabb);
//}
//
//void initBVH(BVH *bvh, Scene *scene) {
//    bvh->node_count = 1 + SPHERE_COUNT / 2;
//    bvh->nodes = AllocN(BVHNode, bvh->node_count);
//    updateBVH(bvh, scene);
//
//    BLAS_line_pixel.color.R = MAX_COLOR_VALUE;
//    BLAS_line_pixel.color.G = 0;
//    BLAS_line_pixel.color.B = MAX_COLOR_VALUE;
//    BLAS_line_pixel.color.A = 0;
//
//    TLAS_line_pixel.color.R = MAX_COLOR_VALUE;
//    TLAS_line_pixel.color.G = MAX_COLOR_VALUE;
//    TLAS_line_pixel.color.B = 0;
//    TLAS_line_pixel.color.A = 0;
//
//    ROOT_line_pixel.color.R = 0;
//    ROOT_line_pixel.color.G = MAX_COLOR_VALUE;
//    ROOT_line_pixel.color.B = MAX_COLOR_VALUE;
//    ROOT_line_pixel.color.A = 0;
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
//void setRayMasksFromBVH(Ray *ray, BVH *bvh) {
//    for (u8 i = 0; i < GEO_TYPE_COUNT; i++) ray->masks.visibility[i] = 0;
//    u8 curr_node_id, next_node_count;
//    BVHNode *node = bvh->nodes;
//
//    u8 next_node_ids[SPHERE_COUNT];
//    u8 curr_node_ids[SPHERE_COUNT];
//    u8 curr_node_count = 2;
//    curr_node_ids[0] = node->children[0];
//    curr_node_ids[1] = node->children[1];
//
//    u8 i, j, mask_index;
//
//    while (curr_node_count) { // Breadth-first_traversal
//        next_node_count = 0;
//
//        for (i = 0; i < curr_node_count; i++) {
//            curr_node_id = curr_node_ids[i];
//            node = &bvh->nodes[curr_node_id];
//            if (hitAABB(&node->aabb, ray)) {
//                for (j = 0; j < 2; j++)
//                if (node->geo_type) {
//                    mask_index = node->geo_type - 1;
//                    ray->masks.visibility[mask_index] |= node->geo_ids[j];
//                } else
//                    next_node_ids[next_node_count++] = node->children[j];
//            }
//        }
//
//        curr_node_count = next_node_count;
//        for (i = 0; i < curr_node_count; i++) curr_node_ids[i] = next_node_ids[i];
//    }
//}
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//void setRayMasksFromBVH2(Ray *ray, BVH *bvh) {
//    for (u8 i = 0; i < GEO_TYPE_COUNT; i++) ray->masks.visibility[i] = 0;
//    u8 child_id, child_bit, next_children_bits;
//    BVHNode *node = bvh->nodes;
//    u8 current_children_bits = node->children_bits;
//
//    while (current_children_bits) { // Breadth-first_traversal
//        next_children_bits = 0;
//        child_bit = 1;
//
//        for (child_id = 0; child_id < 8; child_id++, child_bit <<= (u8)1)
//            if (child_bit & current_children_bits) {
//                node = bvh->nodes + child_id + 1;
//                if (hitAABB(&node->aabb, ray)) {
//                    next_children_bits |= node->children_bits;
//                    if (node->geo_type) {
//                        ray->masks.visibility[node->geo_type - 1] |= node->geo_ids[0];
//                        ray->masks.visibility[node->geo_type - 1] |= node->geo_ids[1];
//                    }
//                }
//            }
//
//        current_children_bits = next_children_bits;
//    }
//}
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//void setRayMasksFromBVH_SoA(Ray *ray, BVH_SoA *bvh) {
//    for (u8 i = 0; i < GEO_TYPE_COUNT; i++) ray->masks.visibility[i] = 0;
//
//    bool hit;
//    hitAABB_init();
//    hitAABB_macro(hit, bvh->root_aabb, (*ray));
//    if (!hit) return;
//
//    u8 child, child_bit, next_children, children = bvh->root.children;
//
//    if (!bvh->root.is_leaf)
//        for (u8 depth = 0; depth < bvh->depth; depth++) {
//            next_children = 0;
//
//            child_bit = 1;
//            for (child = 0; child < 8; child++, child_bit <<= 1)
//                if (children & 1) {
//                    hitAABB_macro(hit, bvh->TLAS_node_aabbs[child], (*ray));
//                    if (hit) next_children |= bvh->TLAS_nodes[child].children;
//
//                }
//
//            children = next_children;
//        }
//
//    child_bit = 1;
//    for (child = 0; child < 8; child++, child_bit <<= 1)
//        if (child_bit & children) {
//            hitAABB_macro(hit, bvh->BLAS_node_aabbs[child], (*ray));
//            if (hit)
//                ray->masks.visibility[bvh->BLAS_nodes[child].geo_type - 1] |= bvh->BLAS_nodes[child].geo_ids;
//        }
//}
//
//void drawBVH_SoA(Camera *camera) {
//    BBox bbox;
//    AABB *aabb = ray_tracer.bvh_soa.BLAS_node_aabbs;
//    for (u8 node_id = 0; node_id < ray_tracer.bvh_soa.BLAS_node_count; node_id++, aabb++) {
//        setBBoxFromAABB(aabb, &bbox);
//        projectBBox(&bbox, camera);
//        drawBBox(&bbox, &GEO_line_pixel);
//    }
//
//    aabb = ray_tracer.bvh_soa.TLAS_node_aabbs;
//    for (u8 node_id = 0; node_id < ray_tracer.bvh_soa.TLAS_node_count; node_id++, aabb++) {
//        setBBoxFromAABB(aabb, &bbox);
//        projectBBox(&bbox, camera);
//        drawBBox(&bbox, &TLAS_line_pixel);
//    }
//    setBBoxFromAABB(&ray_tracer.bvh_soa.root_aabb, &bbox);
//    projectBBox(&bbox, camera);
//    drawBBox(&bbox, &ROOT_line_pixel);
//}
//void drawBVH(Camera *camera) {
//    BBox bbox;
//    BVHNode *bvh_node = ray_tracer.bvh.nodes + 1;
//    for (u8 node_id = 1; node_id < ray_tracer.bvh.node_count; node_id++, bvh_node++) {
//        if (bvh_node->geo_type) {
//            setBBoxFromAABB(&bvh_node->aabb, &bbox);
//            projectBBox(&bbox, camera);
//            drawBBox(&bbox, &BLAS_line_pixel);
//        }
//    }
//    bvh_node = ray_tracer.bvh.nodes + 1;
//    for (u8 node_id = 1; node_id < ray_tracer.bvh.node_count; node_id++, bvh_node++) {
//        if (bvh_node->children_bits) {
//            setBBoxFromAABB(&bvh_node->aabb, &bbox);
//            projectBBox(&bbox, camera);
//            drawBBox(&bbox, &TLAS_line_pixel);
//        }
//    }
//    bvh_node = ray_tracer.bvh.nodes;
//    setBBoxFromAABB(&bvh_node->aabb, &bbox);
//    projectBBox(&bbox, camera);
//    drawBBox(&bbox, &ROOT_line_pixel);
//}