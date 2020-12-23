#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "line.h"

#define BBOX_VERTEX_COUNT 8
#define BBOX_EDGE_COUNT 12

typedef struct {
    vec3 front_top_left,
         front_top_right,
         front_bottom_left,
         front_bottom_right,
         back_top_left,
         back_top_right,
         back_bottom_left,
         back_bottom_right;
} BBoxCorners;
typedef union {
    BBoxCorners corners;
    vec3 buffer[BBOX_VERTEX_COUNT];
} BBoxVertices;

typedef struct {
    vec3 from, to;
    bool is_culled,
         is_clipped;
} Edge;
typedef struct {
    Edge front_top,
         front_bottom,
         front_left,
         front_right,
         back_top,
         back_bottom,
         back_left,
         back_right,
         left_bottom,
         left_top,
         right_bottom,
         right_top;
} BBoxEdgeSides;
typedef union {
    BBoxEdgeSides sides;
    Edge buffer[BBOX_EDGE_COUNT];
} BBoxEdges;

typedef struct {
    BBoxVertices vertex;
    BBoxEdges edge;
} BBox;

void setBBoxEdges(BBox *bbox) {
    bbox->edge.sides.front_top.from    = bbox->vertex.corners.front_top_left;
    bbox->edge.sides.front_top.to      = bbox->vertex.corners.front_top_right;
    bbox->edge.sides.front_bottom.from = bbox->vertex.corners.front_bottom_left;
    bbox->edge.sides.front_bottom.to   = bbox->vertex.corners.front_bottom_right;
    bbox->edge.sides.front_left.from   = bbox->vertex.corners.front_bottom_left;
    bbox->edge.sides.front_left.to     = bbox->vertex.corners.front_top_left;
    bbox->edge.sides.front_right.from  = bbox->vertex.corners.front_bottom_right;
    bbox->edge.sides.front_right.to    = bbox->vertex.corners.front_top_right;

    bbox->edge.sides.back_top.from     = bbox->vertex.corners.back_top_left;
    bbox->edge.sides.back_top.to       = bbox->vertex.corners.back_top_right;
    bbox->edge.sides.back_bottom.from  = bbox->vertex.corners.back_bottom_left;
    bbox->edge.sides.back_bottom.to    = bbox->vertex.corners.back_bottom_right;
    bbox->edge.sides.back_left.from    = bbox->vertex.corners.back_bottom_left;
    bbox->edge.sides.back_left.to      = bbox->vertex.corners.back_top_left;
    bbox->edge.sides.back_right.from   = bbox->vertex.corners.back_bottom_right;
    bbox->edge.sides.back_right.to     = bbox->vertex.corners.back_top_right;

    bbox->edge.sides.left_bottom.from  = bbox->vertex.corners.front_bottom_left;
    bbox->edge.sides.left_bottom.to    = bbox->vertex.corners.back_bottom_left;
    bbox->edge.sides.left_top.from     = bbox->vertex.corners.front_top_left;
    bbox->edge.sides.left_top.to       = bbox->vertex.corners.back_top_left;
    bbox->edge.sides.right_bottom.from = bbox->vertex.corners.front_bottom_right;
    bbox->edge.sides.right_bottom.to   = bbox->vertex.corners.back_bottom_right;
    bbox->edge.sides.right_top.from    = bbox->vertex.corners.front_top_right;
    bbox->edge.sides.right_top.to      = bbox->vertex.corners.back_top_right;
}

void setBBoxFromAABB(AABB *aabb, BBox *bbox) {
    bbox->vertex.corners.front_top_left.x = aabb->min.x;
    bbox->vertex.corners.back_top_left.x = aabb->min.x;
    bbox->vertex.corners.front_bottom_left.x = aabb->min.x;
    bbox->vertex.corners.back_bottom_left.x = aabb->min.x;

    bbox->vertex.corners.front_top_right.x = aabb->max.x;
    bbox->vertex.corners.back_top_right.x = aabb->max.x;
    bbox->vertex.corners.front_bottom_right.x = aabb->max.x;
    bbox->vertex.corners.back_bottom_right.x = aabb->max.x;


    bbox->vertex.corners.front_bottom_left.y = aabb->min.y;
    bbox->vertex.corners.front_bottom_right.y = aabb->min.y;
    bbox->vertex.corners.back_bottom_left.y = aabb->min.y;
    bbox->vertex.corners.back_bottom_right.y = aabb->min.y;

    bbox->vertex.corners.front_top_left.y = aabb->max.y;
    bbox->vertex.corners.front_top_right.y = aabb->max.y;
    bbox->vertex.corners.back_top_left.y = aabb->max.y;
    bbox->vertex.corners.back_top_right.y = aabb->max.y;


    bbox->vertex.corners.front_top_left.z = aabb->min.z;
    bbox->vertex.corners.front_top_right.z = aabb->min.z;
    bbox->vertex.corners.front_bottom_left.z = aabb->min.z;
    bbox->vertex.corners.front_bottom_right.z = aabb->min.z;

    bbox->vertex.corners.back_top_left.z = aabb->max.z;
    bbox->vertex.corners.back_top_right.z = aabb->max.z;
    bbox->vertex.corners.back_bottom_left.z = aabb->max.z;
    bbox->vertex.corners.back_bottom_right.z = aabb->max.z;

    setBBoxEdges(bbox);
}

inline void projectBBox(BBox *bbox, Camera *camera) {
    // Transform vertex positions from world-space to view-space:
    for (u8 i = 0; i < BBOX_VERTEX_COUNT; i++) {
        isubVec3(&bbox->vertex.buffer[i], &camera->transform.position);
        imulVec3Mat3(&bbox->vertex.buffer[i], &camera->transform.rotation_matrix_inverted);
    }

    // Distribute transformed vertex positions to edges:
    setBBoxEdges(bbox);

    // Transform vertex positions of edges from view-space to screen-space (w/ culling and clipping):
    f32 x_factor = camera->focal_length;
    f32 y_factor = camera->focal_length * frame_buffer.width_over_height;
    for (u8 i = 0; i < BBOX_EDGE_COUNT; i++)
        projectEdge(&bbox->edge.buffer[i].from,
                    &bbox->edge.buffer[i].to,
                    x_factor,
                    y_factor);
}

inline void drawBBox(BBox *bbox, Pixel *pixel) {
    for (u8 i = 0; i < BBOX_EDGE_COUNT; i++)
        drawLine2D((i32)bbox->edge.buffer[i].from.x,
                   (i32)bbox->edge.buffer[i].from.y,
                   (i32)bbox->edge.buffer[i].to.x,
                   (i32)bbox->edge.buffer[i].to.y,
                   pixel);
}