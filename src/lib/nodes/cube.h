#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"


void updateCubePosition(Cube *cube, vec3 *position) {
    vec3 movement;
    subVec3(position, &cube->node.position, &movement);
    for (u8 i = 0; i < 8; i++) iaddVec3(&cube->vertices[i], &movement);
    cube->node.position = *position;
}

void updateCubeMatrices(Cube *cube) {
    mat3 scale;
    setMat3ToIdentity(&scale);
    scale.X.x = scale.Y.y = 1 / (2 * cube->node.radius / SQRT3);

    for (u8 i = 0; i < 6; i++) {
        transposeMat3(&cube->tangent_to_world[i], &cube->world_to_tangent[i]);
        imulMat3(&cube->world_to_tangent[i], &scale);
    }
}

void updateCubeRadius(Cube *cube, f32 radius) {
    f32 factor = radius / cube->node.radius;
    for (u8 i = 0; i < 8; i++) {
        isubVec3(&cube->vertices[i], &cube->node.position);
        iscaleVec3(&cube->vertices[i], factor);
        iaddVec3(&cube->vertices[i], &cube->node.position);
    }
    cube->node.radius = radius;
    updateCubeMatrices(cube);
}

void initCube(Cube *cube, f32 radius, vec3 *vertices, QuadIndices *indices) {
    cube->node.radius = radius;

    // Init vertex positions:
    for (u8 i = 0; i < 8; i++) cube->vertices[i] = vertices[i];

    // Init quads:
    for (u8 i = 0; i < 6; i++) {
        subVec3(&cube->vertices[indices[i].v4],
                &cube->vertices[indices[i].v1],
                &cube->tangent_to_world[i].X);
        subVec3(&cube->vertices[indices[i].v2],
                &cube->vertices[indices[i].v1],
                &cube->tangent_to_world[i].Y);

        crossVec3(&cube->tangent_to_world[i].X,
                  &cube->tangent_to_world[i].Y,
                  &cube->tangent_to_world[i].Z);

        norm3(&cube->tangent_to_world[i].X);
        norm3(&cube->tangent_to_world[i].Y);
        norm3(&cube->tangent_to_world[i].Z);

        invertVec3(&cube->tangent_to_world[i].Z);
    }

    f32 half_cube_edge  = cube->node.radius / SQRT3;
    vec3 offset;
    fillVec3(&offset, half_cube_edge);

    vec3 *vertex_poisition = cube->vertices;
    for (u8 i = 0; i < 8; i++, vertex_poisition++) {
        iscaleVec3(vertex_poisition, half_cube_edge + half_cube_edge);
        isubVec3(vertex_poisition, &offset);
    }

    updateCubeMatrices(cube);
}
//
//void initCube(Cube* cube) {
//    vec3* vertex = cube->vertices;
//    vec3* front_top_left_vertex = vertex++;
//    vec3* front_top_right_vertex = vertex++;
//    vec3* front_bottom_left_vertex = vertex++;
//    vec3* front_bottom_right_vertex = vertex++;
//    vec3* back_top_left_vertex = vertex++;
//    vec3* back_top_right_vertex = vertex++;
//    vec3* back_bottom_left_vertex = vertex++;
//    vec3* back_bottom_right_vertex = vertex;
//
//    front_top_left_vertex->x = front_bottom_left_vertex->x = back_top_left_vertex->x = back_bottom_left_vertex->x = -0.5f;
//    front_top_right_vertex->x = front_bottom_right_vertex->x = back_top_right_vertex->x = back_bottom_right_vertex->x = 0.5f;
//
//    front_top_left_vertex->y = front_top_right_vertex->y = back_top_left_vertex->y = back_top_right_vertex->y = 0.5f;
//    front_bottom_left_vertex->y = front_bottom_left_vertex->y = back_bottom_left_vertex->y = back_bottom_right_vertex->y = -0.5f;
//
//    front_top_left_vertex->z = front_top_right_vertex->z = front_bottom_left_vertex->z = front_bottom_right_vertex->z = -0.5f;
//    back_top_left_vertex->z = back_top_right_vertex->z = back_bottom_left_vertex->z = back_bottom_right_vertex->z = 0.5f;
//
//    vec3 up = {0, 1, 0};
//    vec3 down = {0, -1, 0};
//    vec3 left = {-1, 0, 0};
//    vec3 right = {1, 0, 0};
//    vec3 forward = {0, 0, 1};
//    vec3 backward = {0, 0, -1};
//
//    Triangle* triangle = cube->triangles;
//    Triangle* front_top_left_triangle = triangle++;
//    Triangle* front_bottom_right_triangle = triangle++;
//    Triangle* right_top_left_triangle = triangle++;
//    Triangle* right_bottom_right_triangle = triangle++;
//    Triangle* back_top_left_triangle = triangle++;
//    Triangle* back_bottom_right_triangle = triangle++;
//    Triangle* left_top_left_triangle = triangle++;
//    Triangle* left_bottom_right_triangle = triangle++;
//    Triangle* top_top_left_triangle = triangle++;
//    Triangle* top_bottom_right_triangle = triangle++;
//    Triangle* bottom_top_left_triangle = triangle++;
//    Triangle* bottom_bottom_right_triangle = triangle;
//
//    // Front quad:
//    front_top_left_triangle->v1 = (u8)(front_top_left_vertex - cube->vertices);
//    front_top_left_triangle->v2 = (u8)(front_bottom_left_vertex - cube->vertices);
//    front_top_left_triangle->v3 = (u8)(front_top_right_vertex - cube->vertices);
//    front_top_left_triangle->tangent_to_world.X = right;
//    front_top_left_triangle->tangent_to_world.Y = down;
//    front_top_left_triangle->tangent_to_world.Z = backward;
//    transposeMat3(&front_top_left_triangle->tangent_to_world,
//                  &front_top_left_triangle->world_to_tangent);
//
//    front_bottom_right_triangle->v1 = (u8)(front_bottom_right_vertex - cube->vertices);
//    front_bottom_right_triangle->v2 = (u8)(front_top_right_vertex - cube->vertices);
//    front_bottom_right_triangle->v3 = (u8)(front_bottom_left_vertex - cube->vertices);
//    front_bottom_right_triangle->tangent_to_world.X = left;
//    front_bottom_right_triangle->tangent_to_world.Y = up;
//    front_bottom_right_triangle->tangent_to_world.Z = backward;
//    transposeMat3(&front_bottom_right_triangle->tangent_to_world,
//                  &front_bottom_right_triangle->world_to_tangent);
//
//    // Right quad:
//    right_top_left_triangle->v1 = (u8)(front_top_right_vertex - cube->vertices);
//    right_top_left_triangle->v2 = (u8)(front_bottom_right_vertex - cube->vertices);
//    right_top_left_triangle->v3 = (u8)(back_top_right_vertex - cube->vertices);
//    right_top_left_triangle->tangent_to_world.X = forward;
//    right_top_left_triangle->tangent_to_world.Y = down;
//    right_top_left_triangle->tangent_to_world.Z = right;
//    transposeMat3(&right_top_left_triangle->tangent_to_world,
//                  &right_top_left_triangle->world_to_tangent);
//
//    right_bottom_right_triangle->v1 = (u8)(back_bottom_right_vertex - cube->vertices);
//    right_bottom_right_triangle->v2 = (u8)(back_top_right_vertex - cube->vertices);
//    right_bottom_right_triangle->v3 = (u8)(front_bottom_right_vertex - cube->vertices);
//    right_bottom_right_triangle->tangent_to_world.X = backward;
//    right_bottom_right_triangle->tangent_to_world.Y = up;
//    right_bottom_right_triangle->tangent_to_world.Z = right;
//    transposeMat3(&right_bottom_right_triangle->tangent_to_world,
//                  &right_bottom_right_triangle->world_to_tangent);
//
//    // Back quad:
//    back_top_left_triangle->v1 = (u8)(back_top_right_vertex - cube->vertices);
//    back_top_left_triangle->v2 = (u8)(back_bottom_right_vertex - cube->vertices);
//    back_top_left_triangle->v3 = (u8)(back_top_left_vertex - cube->vertices);
//    back_top_left_triangle->tangent_to_world.X = left;
//    back_top_left_triangle->tangent_to_world.Y = down;
//    back_top_left_triangle->tangent_to_world.Z = forward;
//    transposeMat3(&back_top_left_triangle->tangent_to_world,
//                  &back_top_left_triangle->world_to_tangent);
//
//    back_bottom_right_triangle->v1 = (u8)(back_bottom_left_vertex - cube->vertices);
//    back_bottom_right_triangle->v2 = (u8)(back_top_left_vertex - cube->vertices);
//    back_bottom_right_triangle->v3 = (u8)(back_bottom_right_vertex - cube->vertices);
//    back_bottom_right_triangle->tangent_to_world.X = right;
//    back_bottom_right_triangle->tangent_to_world.Y = up;
//    back_bottom_right_triangle->tangent_to_world.Z = forward;
//    transposeMat3(&back_bottom_right_triangle->tangent_to_world,
//                  &back_bottom_right_triangle->world_to_tangent);
//
//    // Left quad:
//    left_top_left_triangle->v1 = (u8)(back_top_left_vertex - cube->vertices);
//    left_top_left_triangle->v2 = (u8)(back_bottom_left_vertex - cube->vertices);
//    left_top_left_triangle->v3 = (u8)(front_top_left_vertex - cube->vertices);
//    left_top_left_triangle->tangent_to_world.X = backward;
//    left_top_left_triangle->tangent_to_world.Y = down;
//    left_top_left_triangle->tangent_to_world.Z = left;
//    transposeMat3(&left_top_left_triangle->tangent_to_world,
//                  &left_top_left_triangle->world_to_tangent);
//
//    left_bottom_right_triangle->v1 = (u8)(front_bottom_left_vertex - cube->vertices);
//    left_bottom_right_triangle->v2 = (u8)(front_top_left_vertex - cube->vertices);
//    left_bottom_right_triangle->v3 = (u8)(back_bottom_left_vertex - cube->vertices);
//    left_bottom_right_triangle->tangent_to_world.X = forward;
//    left_bottom_right_triangle->tangent_to_world.Y = up;
//    left_bottom_right_triangle->tangent_to_world.Z = left;
//    transposeMat3(&left_bottom_right_triangle->tangent_to_world,
//                  &left_bottom_right_triangle->world_to_tangent);
//
//    // Top quad:
//    top_top_left_triangle->v1 = (u8)(back_top_left_vertex - cube->vertices);
//    top_top_left_triangle->v2 = (u8)(front_top_left_vertex - cube->vertices);
//    top_top_left_triangle->v3 = (u8)(back_top_right_vertex - cube->vertices);
//    top_top_left_triangle->tangent_to_world.X = right;
//    top_top_left_triangle->tangent_to_world.Y = backward;
//    top_top_left_triangle->tangent_to_world.Z = up;
//    transposeMat3(&top_top_left_triangle->tangent_to_world,
//                  &top_top_left_triangle->world_to_tangent);
//
//    top_bottom_right_triangle->v1 = (u8)(front_top_right_vertex - cube->vertices);
//    top_bottom_right_triangle->v2 = (u8)(back_top_right_vertex - cube->vertices);
//    top_bottom_right_triangle->v3 = (u8)(front_top_left_vertex - cube->vertices);
//    top_bottom_right_triangle->tangent_to_world.X = left;
//    top_bottom_right_triangle->tangent_to_world.Y = forward;
//    top_bottom_right_triangle->tangent_to_world.Z = up;
//    transposeMat3(&top_bottom_right_triangle->tangent_to_world,
//                  &top_bottom_right_triangle->world_to_tangent);
//
//    // Bottom quad:
//    bottom_top_left_triangle->v1 = (u8)(front_bottom_left_vertex - cube->vertices);
//    bottom_top_left_triangle->v2 = (u8)(back_bottom_left_vertex - cube->vertices);
//    bottom_top_left_triangle->v3 = (u8)(front_bottom_right_vertex - cube->vertices);
//    bottom_top_left_triangle->tangent_to_world.X = right;
//    bottom_top_left_triangle->tangent_to_world.Y = forward;
//    bottom_top_left_triangle->tangent_to_world.Z = down;
//    transposeMat3(&bottom_top_left_triangle->tangent_to_world,
//                  &bottom_top_left_triangle->world_to_tangent);
//
//    bottom_bottom_right_triangle->v1 = (u8)(back_bottom_right_vertex - cube->vertices);
//    bottom_bottom_right_triangle->v2 = (u8)(front_bottom_right_vertex - cube->vertices);
//    bottom_bottom_right_triangle->v3 = (u8)(back_bottom_left_vertex - cube->vertices);
//    bottom_bottom_right_triangle->tangent_to_world.X = left;
//    bottom_bottom_right_triangle->tangent_to_world.Y = backward;
//    bottom_bottom_right_triangle->tangent_to_world.Z = down;
//    transposeMat3(&bottom_bottom_right_triangle->tangent_to_world,
//                  &bottom_bottom_right_triangle->world_to_tangent);
//
//    setMat3ToIdentity(&cube->rotation_matrix);
//}