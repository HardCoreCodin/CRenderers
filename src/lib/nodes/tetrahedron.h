#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

mat3 tetrahedron_tangent_to_world[4];
bool _tetrahedron_tangent_to_world_initialized = false;

void initTetrahedron(Tetrahedron *tet) {
    Triangle *top_triangle   = &tet->triangles[0];
    Triangle *front_triangle = &tet->triangles[1];
    Triangle *right_triangle = &tet->triangles[2];
    Triangle *left_triangle  = &tet->triangles[3];
    vec3 *bottom_vertex    = &tet->vertices[0];
    vec3 *top_left_vertex  = &tet->vertices[1];
    vec3 *top_right_vertex = &tet->vertices[2];
    vec3 *top_back_vertex  = &tet->vertices[3];

    bottom_vertex->x = 0;
    bottom_vertex->y = 0;
    bottom_vertex->z = 0;

    top_left_vertex->x = -0.5f;
    top_left_vertex->y = SQRT_OF_TWO_THIRDS;
    top_left_vertex->z = -SQRT_OF_THREE_OVER_SIX;

    top_right_vertex->x = 0.5f;
    top_right_vertex->y = SQRT_OF_TWO_THIRDS;
    top_right_vertex->z = -SQRT_OF_THREE_OVER_SIX;

    top_back_vertex->x = 0;
    top_back_vertex->y = SQRT_OF_TWO_THIRDS;
    top_back_vertex->z = SQRT_OF_THREE_OVER_THREE;

    top_triangle->p1 = top_left_vertex;
    top_triangle->p2 = top_right_vertex;
    top_triangle->p3 = top_back_vertex;

    front_triangle->p1 = bottom_vertex;
    front_triangle->p2 = top_right_vertex;
    front_triangle->p3 = top_left_vertex;

    right_triangle->p1 = bottom_vertex;
    right_triangle->p2 = top_back_vertex;
    right_triangle->p3 = top_right_vertex;

    left_triangle->p1 = bottom_vertex;
    left_triangle->p2 = top_left_vertex;
    left_triangle->p3 = top_back_vertex;

    mat3 *matrix;
//    if (!_tetrahedron_tangent_to_world_initialized) {
        // Top triangle:
        matrix = &tetrahedron_tangent_to_world[0];
        subVec3(top_right_vertex, top_left_vertex, &matrix->X); norm3(&matrix->X);
        subVec3(top_back_vertex, top_left_vertex, &matrix->Y); norm3(&matrix->Y);
        crossVec3(&matrix->X, &matrix->Y, &matrix->Z); norm3(&matrix->Z);
        crossVec3(&matrix->Z, &matrix->X, &matrix->Y); norm3(&matrix->Y);

        // Front triangle:
        matrix = &tetrahedron_tangent_to_world[1];
        matrix->X = *top_right_vertex;  norm3(&matrix->X);
        matrix->Y = *top_left_vertex;   norm3(&matrix->Y);
        crossVec3(&matrix->X, &matrix->Y, &matrix->Z); norm3(&matrix->Z);
        crossVec3(&matrix->Z, &matrix->X, &matrix->Y); norm3(&matrix->Y);

        // Right triangle:
        matrix = &tetrahedron_tangent_to_world[2];
        matrix->X = *top_left_vertex;  norm3(&matrix->X);
        matrix->Y = *top_back_vertex;  norm3(&matrix->Y);
        crossVec3(&matrix->X, &matrix->Y, &matrix->Z); norm3(&matrix->Z);
        crossVec3(&matrix->Z, &matrix->X, &matrix->Y); norm3(&matrix->Y);

        // Left triangle:
        matrix = &tetrahedron_tangent_to_world[3];
        matrix->X = *top_back_vertex;  norm3(&matrix->X);
        matrix->Y = *top_right_vertex; norm3(&matrix->Y);
        crossVec3(&matrix->X, &matrix->Y, &matrix->Z); norm3(&matrix->Z);
        crossVec3(&matrix->Z, &matrix->X, &matrix->Y); norm3(&matrix->Y);
//    }
    top_triangle->tangent_to_world   = tetrahedron_tangent_to_world[0];
    front_triangle->tangent_to_world = tetrahedron_tangent_to_world[1];
    right_triangle->tangent_to_world = tetrahedron_tangent_to_world[2];
    left_triangle->tangent_to_world  = tetrahedron_tangent_to_world[3];

    transposeMat3(  &top_triangle->tangent_to_world,   &top_triangle->world_to_tangent);
    transposeMat3(&front_triangle->tangent_to_world, &front_triangle->world_to_tangent);
    transposeMat3(&right_triangle->tangent_to_world, &right_triangle->world_to_tangent);
    transposeMat3( &left_triangle->tangent_to_world,  &left_triangle->world_to_tangent);

    top_triangle->normal   = &top_triangle->tangent_to_world.Z;
    front_triangle->normal = &front_triangle->tangent_to_world.Z;
    right_triangle->normal = &right_triangle->tangent_to_world.Z;
    left_triangle->normal  = &left_triangle->tangent_to_world.Z;


    setMat3ToIdentity(&tet->rotation_matrix);
}