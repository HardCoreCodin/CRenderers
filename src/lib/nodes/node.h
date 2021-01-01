#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"

inline void setNodePosition(Node *node, vec3 *position) {
    u8 vertex_count;
    vec3 *vertex_positions;

    switch (node->geo.type) {
        case GeoTypeCube:
            vertex_count = 8;
            vertex_positions = ((Cube*)node)->vertices;
            break;

        case GeoTypeTetrahedron:
            vertex_count = 4;
            vertex_positions = ((Tetrahedron*)node)->vertices;
            break;
    }

    vec3 movement;
    subVec3(position, &node->position, &movement);
    for (u8 i = 0; i < vertex_count; i++) iaddVec3(vertex_positions + i, &movement);
    node->position = *position;
}

inline void setNodeTangentMatrices(Node *node) {
    mat3 transform;
    u8 face_count;
    mat3 *tangent_to_world, *world_to_tangent;

    switch (node->geo.type) {
        case GeoTypeCube:
            face_count = 6;
            tangent_to_world = ((Cube*)node)->tangent_to_world;
            world_to_tangent = ((Cube*)node)->world_to_tangent;

            setMat3ToIdentity(&transform);
            transform.X.x = transform.Y.y = 1 / (2 * node->radius / SQRT3);
            break;
        case GeoTypeTetrahedron:
            face_count = 4;
            tangent_to_world = ((Tetrahedron*)node)->tangent_to_world;
            world_to_tangent = ((Tetrahedron*)node)->world_to_tangent;

            mat3 scale, skew;
            setMat3ToIdentity(&skew);
            setMat3ToIdentity(&scale);
            scale.X.x = 1 / (node->radius * 2*SQRT2/SQRT3);
            scale.Y.y = 1 / (node->radius * SQRT2);
            skew.Y.x = -0.5;
            mulMat3(&scale, &skew, &transform);
            break;
    }

    for (u8 i = 0; i < face_count; i++) {
        transposeMat3(tangent_to_world + i, world_to_tangent + i);
        imulMat3(world_to_tangent + i, &transform);
    }
}

inline void rotateNode(Node *node, mat3 *rotation) {
    u8 vertex_count, face_count;
    vec3 *vertex_positions;
    mat3 *tangent_to_world;

    switch (node->geo.type) {
        case GeoTypeCube:
            face_count = 6;
            vertex_count = 8;
            vertex_positions = ((Cube*)node)->vertices;
            tangent_to_world = ((Cube*)node)->tangent_to_world;
            break;

        case GeoTypeTetrahedron:
            face_count = 4;
            vertex_count = 4;
            vertex_positions = ((Tetrahedron*)node)->vertices;
            tangent_to_world = ((Tetrahedron*)node)->tangent_to_world;
            break;
    }

    vec3 *position = &node->position;
    vec3 *vertex = vertex_positions;
    for (u8 i = 0; i < vertex_count; i++, vertex++) {
        isubVec3(vertex, position);
        imulVec3Mat3(vertex, rotation);
        iaddVec3(vertex, position);
    }

    for (u8 i = 0; i < face_count; i++) imulMat3(tangent_to_world + i, rotation);

    setNodeTangentMatrices(node);
}

inline void setNodeRadius(Node *node, f32 radius) {
    u8 vertex_count;
    vec3 *vertex_positions;

    switch (node->geo.type) {
        case GeoTypeCube:
            vertex_count = 8;
            vertex_positions = ((Cube*)node)->vertices;
            break;

        case GeoTypeTetrahedron:
            vertex_count = 4;
            vertex_positions = ((Tetrahedron*)node)->vertices;
            break;
    }

    vec3 *vertex = vertex_positions;
    vec3 *position = &node->position;
    f32 factor = radius / node->radius;
    for (u8 i = 0; i < vertex_count; i++, vertex++) {
        isubVec3(vertex, position);
        iscaleVec3(vertex, factor);
        iaddVec3(vertex, position);
    }
    node->radius = radius;
    setNodeTangentMatrices(node);
}

void initNode(Node *node, f32 radius) {
    node->radius = radius;

    u8 vertex_count, face_count;
    vec3 *vertex_positions, *initial_vertex_positions;
    mat3 *tangent_to_world;
    Indices *indices;

    switch (node->geo.type) {
        case GeoTypeCube:
            face_count = 6;
            vertex_count = 8;
            vertex_positions = ((Cube*)node)->vertices;
            tangent_to_world = ((Cube*)node)->tangent_to_world;
            indices = cube_indices;
            initial_vertex_positions = cube_initial_vertex_positions;
            break;

        case GeoTypeTetrahedron:
            face_count = 4;
            vertex_count = 4;
            vertex_positions = ((Tetrahedron*)node)->vertices;
            tangent_to_world = ((Tetrahedron*)node)->tangent_to_world;
            indices = tetrahedron_indices;
            initial_vertex_positions = tetrahedron_initial_vertex_positions;
            break;
    }

    // Init vertex positions:
    for (u8 i = 0; i < vertex_count; i++) vertex_positions[i] = initial_vertex_positions[i];

    // Init triangles:
    for (u8 i = 0; i < face_count; i++) {
        subVec3(vertex_positions + (node->geo.type == GeoTypeCube ? indices[i].v4 : indices[i].v3),
                vertex_positions + indices[i].v1,
                &tangent_to_world[i].X);
        subVec3(vertex_positions + indices[i].v2,
                vertex_positions + indices[i].v1,
                &tangent_to_world[i].Y);

        crossVec3(&tangent_to_world[i].X,
                  &tangent_to_world[i].Y,
                  &tangent_to_world[i].Z);

        if (node->geo.type == GeoTypeTetrahedron)
            crossVec3(&tangent_to_world[i].Z,
                      &tangent_to_world[i].X,
                      &tangent_to_world[i].Y);

        norm3(&tangent_to_world[i].X);
        norm3(&tangent_to_world[i].Y);
        norm3(&tangent_to_world[i].Z);

        invertVec3(&tangent_to_world[i].Z);
    }

    f32 half_cube_edge  = radius / SQRT3;
    vec3 offset;
    fillVec3(&offset, half_cube_edge);

    vec3 *vertex_poisition = vertex_positions;
    for (u8 i = 0; i < vertex_count; i++, vertex_poisition++) {
        iscaleVec3(vertex_poisition, half_cube_edge + half_cube_edge);
        isubVec3(vertex_poisition, &offset);
    }

    setNodeTangentMatrices(node);
}