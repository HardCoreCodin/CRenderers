#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"

void initCube(Cube* cube) {
    vec3* vertex = cube->vertices;
    vec3* front_top_left_vertex = vertex++;
    vec3* front_top_right_vertex = vertex++;
    vec3* front_bottom_left_vertex = vertex++;
    vec3* front_bottom_right_vertex = vertex++;
    vec3* back_top_left_vertex = vertex++;
    vec3* back_top_right_vertex = vertex++;
    vec3* back_bottom_left_vertex = vertex++;
    vec3* back_bottom_right_vertex = vertex;

    front_top_left_vertex->x = front_bottom_left_vertex->x = back_top_left_vertex->x = back_bottom_left_vertex->x = -0.5f;
    front_top_right_vertex->x = front_bottom_right_vertex->x = back_top_right_vertex->x = back_bottom_right_vertex->x = 0.5f;

    front_top_left_vertex->y = front_top_right_vertex->y = back_top_left_vertex->y = back_top_right_vertex->y = 0.5f;
    front_bottom_left_vertex->y = front_bottom_left_vertex->y = back_bottom_left_vertex->y = back_bottom_right_vertex->y = -0.5f;

    front_top_left_vertex->z = front_top_right_vertex->z = front_bottom_left_vertex->z = front_bottom_right_vertex->z = -0.5f;
    back_top_left_vertex->z = back_top_right_vertex->z = back_bottom_left_vertex->z = back_bottom_right_vertex->z = 0.5f;

    vec3 up = {0, 1, 0};
    vec3 down = {0, -1, 0};
    vec3 left = {-1, 0, 0};
    vec3 right = {1, 0, 0};
    vec3 forward = {0, 0, 1};
    vec3 backward = {0, 0, -1};

    Triangle* triangle = cube->triangles;
    Triangle* front_top_left_triangle = triangle++;
    Triangle* front_bottom_right_triangle = triangle++;
    Triangle* right_top_left_triangle = triangle++;
    Triangle* right_bottom_right_triangle = triangle++;
    Triangle* back_top_left_triangle = triangle++;
    Triangle* back_bottom_right_triangle = triangle++;
    Triangle* left_top_left_triangle = triangle++;
    Triangle* left_bottom_right_triangle = triangle++;
    Triangle* top_top_left_triangle = triangle++;
    Triangle* top_bottom_right_triangle = triangle++;
    Triangle* bottom_top_left_triangle = triangle++;
    Triangle* bottom_bottom_right_triangle = triangle;

    // Front quad:
    front_top_left_triangle->p1 = front_top_left_vertex - cube->vertices;
    front_top_left_triangle->p2 = front_bottom_left_vertex - cube->vertices;
    front_top_left_triangle->p3 = front_top_right_vertex - cube->vertices;
    front_top_left_triangle->tangent_to_world.X = right;
    front_top_left_triangle->tangent_to_world.Y = down;
    front_top_left_triangle->tangent_to_world.Z = backward;
    transposeMat3(&front_top_left_triangle->tangent_to_world,
                  &front_top_left_triangle->world_to_tangent);

    front_bottom_right_triangle->p1 = front_bottom_right_vertex - cube->vertices;
    front_bottom_right_triangle->p2 = front_top_right_vertex - cube->vertices;
    front_bottom_right_triangle->p3 = front_bottom_left_vertex - cube->vertices;
    front_bottom_right_triangle->tangent_to_world.X = left;
    front_bottom_right_triangle->tangent_to_world.Y = up;
    front_bottom_right_triangle->tangent_to_world.Z = backward;
    transposeMat3(&front_bottom_right_triangle->tangent_to_world,
                  &front_bottom_right_triangle->world_to_tangent);

    // Right quad:
    right_top_left_triangle->p1 = front_top_right_vertex - cube->vertices;
    right_top_left_triangle->p2 = front_bottom_right_vertex - cube->vertices;
    right_top_left_triangle->p3 = back_top_right_vertex - cube->vertices;
    right_top_left_triangle->tangent_to_world.X = forward;
    right_top_left_triangle->tangent_to_world.Y = down;
    right_top_left_triangle->tangent_to_world.Z = right;
    transposeMat3(&right_top_left_triangle->tangent_to_world,
                  &right_top_left_triangle->world_to_tangent);

    right_bottom_right_triangle->p1 = back_bottom_right_vertex - cube->vertices;
    right_bottom_right_triangle->p2 = back_top_right_vertex - cube->vertices;
    right_bottom_right_triangle->p3 = front_bottom_right_vertex - cube->vertices;
    right_bottom_right_triangle->tangent_to_world.X = backward;
    right_bottom_right_triangle->tangent_to_world.Y = up;
    right_bottom_right_triangle->tangent_to_world.Z = right;
    transposeMat3(&right_bottom_right_triangle->tangent_to_world,
                  &right_bottom_right_triangle->world_to_tangent);

    // Back quad:
    back_top_left_triangle->p1 = back_top_right_vertex - cube->vertices;
    back_top_left_triangle->p2 = back_bottom_right_vertex - cube->vertices;
    back_top_left_triangle->p3 = back_top_left_vertex - cube->vertices;
    back_top_left_triangle->tangent_to_world.X = left;
    back_top_left_triangle->tangent_to_world.Y = down;
    back_top_left_triangle->tangent_to_world.Z = forward;
    transposeMat3(&back_top_left_triangle->tangent_to_world,
                  &back_top_left_triangle->world_to_tangent);

    back_bottom_right_triangle->p1 = back_bottom_left_vertex - cube->vertices;
    back_bottom_right_triangle->p2 = back_top_left_vertex - cube->vertices;
    back_bottom_right_triangle->p3 = back_bottom_right_vertex - cube->vertices;
    back_bottom_right_triangle->tangent_to_world.X = right;
    back_bottom_right_triangle->tangent_to_world.Y = up;
    back_bottom_right_triangle->tangent_to_world.Z = forward;
    transposeMat3(&back_bottom_right_triangle->tangent_to_world,
                  &back_bottom_right_triangle->world_to_tangent);

    // Left quad:
    left_top_left_triangle->p1 = back_top_left_vertex - cube->vertices;
    left_top_left_triangle->p2 = back_bottom_left_vertex - cube->vertices;
    left_top_left_triangle->p3 = front_top_left_vertex - cube->vertices;
    left_top_left_triangle->tangent_to_world.X = backward;
    left_top_left_triangle->tangent_to_world.Y = down;
    left_top_left_triangle->tangent_to_world.Z = left;
    transposeMat3(&left_top_left_triangle->tangent_to_world,
                  &left_top_left_triangle->world_to_tangent);

    left_bottom_right_triangle->p1 = front_bottom_left_vertex - cube->vertices;
    left_bottom_right_triangle->p2 = front_top_left_vertex - cube->vertices;
    left_bottom_right_triangle->p3 = back_bottom_left_vertex - cube->vertices;
    left_bottom_right_triangle->tangent_to_world.X = forward;
    left_bottom_right_triangle->tangent_to_world.Y = up;
    left_bottom_right_triangle->tangent_to_world.Z = left;
    transposeMat3(&left_bottom_right_triangle->tangent_to_world,
                  &left_bottom_right_triangle->world_to_tangent);

    // Top quad:
    top_top_left_triangle->p1 = back_top_left_vertex - cube->vertices;
    top_top_left_triangle->p2 = front_top_left_vertex - cube->vertices;
    top_top_left_triangle->p3 = back_top_right_vertex - cube->vertices;
    top_top_left_triangle->tangent_to_world.X = right;
    top_top_left_triangle->tangent_to_world.Y = backward;
    top_top_left_triangle->tangent_to_world.Z = up;
    transposeMat3(&top_top_left_triangle->tangent_to_world,
                  &top_top_left_triangle->world_to_tangent);

    top_bottom_right_triangle->p1 = front_top_right_vertex - cube->vertices;
    top_bottom_right_triangle->p2 = back_top_right_vertex - cube->vertices;
    top_bottom_right_triangle->p3 = front_top_left_vertex - cube->vertices;
    top_bottom_right_triangle->tangent_to_world.X = left;
    top_bottom_right_triangle->tangent_to_world.Y = forward;
    top_bottom_right_triangle->tangent_to_world.Z = up;
    transposeMat3(&top_bottom_right_triangle->tangent_to_world,
                  &top_bottom_right_triangle->world_to_tangent);

    // Bottom quad:
    bottom_top_left_triangle->p1 = front_bottom_left_vertex - cube->vertices;
    bottom_top_left_triangle->p2 = back_bottom_left_vertex - cube->vertices;
    bottom_top_left_triangle->p3 = front_bottom_right_vertex - cube->vertices;
    bottom_top_left_triangle->tangent_to_world.X = right;
    bottom_top_left_triangle->tangent_to_world.Y = forward;
    bottom_top_left_triangle->tangent_to_world.Z = down;
    transposeMat3(&bottom_top_left_triangle->tangent_to_world,
                  &bottom_top_left_triangle->world_to_tangent);

    bottom_bottom_right_triangle->p1 = back_bottom_right_vertex - cube->vertices;
    bottom_bottom_right_triangle->p2 = front_bottom_right_vertex - cube->vertices;
    bottom_bottom_right_triangle->p3 = back_bottom_left_vertex - cube->vertices;
    bottom_bottom_right_triangle->tangent_to_world.X = left;
    bottom_bottom_right_triangle->tangent_to_world.Y = backward;
    bottom_bottom_right_triangle->tangent_to_world.Z = down;
    transposeMat3(&bottom_bottom_right_triangle->tangent_to_world,
                  &bottom_bottom_right_triangle->world_to_tangent);

    setMat3ToIdentity(&cube->rotation_matrix);
}