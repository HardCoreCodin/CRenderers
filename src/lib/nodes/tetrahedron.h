#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"


void initTetrahedron(Tetrahedron *tet) {
    initXform3(&tet->xform);

    Triangle *triangle = tet->triangles;
    Triangle *top_triangle   = triangle++;
    Triangle *front_triangle = triangle++;
    Triangle *right_triangle = triangle++;
    Triangle *left_triangle  = triangle++;

    vec3 *vertex = tet->vertices;
    vec3 *bottom_vertex    = vertex++;
    vec3 *top_left_vertex  = vertex++;
    vec3 *top_right_vertex = vertex++;
    vec3 *top_back_vertex  = vertex++;

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

    top_triangle->p1 = top_left_vertex - tet->vertices;
    top_triangle->p2 = top_right_vertex - tet->vertices;
    top_triangle->p3 = top_back_vertex - tet->vertices;

    front_triangle->p1 = bottom_vertex - tet->vertices;
    front_triangle->p2 = top_right_vertex - tet->vertices;
    front_triangle->p3 = top_left_vertex - tet->vertices;

    right_triangle->p1 = bottom_vertex - tet->vertices;
    right_triangle->p2 = top_back_vertex - tet->vertices;
    right_triangle->p3 = top_right_vertex - tet->vertices;

    left_triangle->p1 = bottom_vertex - tet->vertices;
    left_triangle->p2 = top_left_vertex - tet->vertices;
    left_triangle->p3 = top_back_vertex - tet->vertices;

    triangle = tet->triangles;
    for (u8 i = 0; i < 4; i++, triangle++) {
        subVec3(&tet->vertices[triangle->p3],
                &tet->vertices[triangle->p1],
                &triangle->tangent_to_world.X);
        subVec3(&tet->vertices[triangle->p2],
                &tet->vertices[triangle->p1],
                &triangle->tangent_to_world.Y);

        norm3(&triangle->tangent_to_world.X);
        norm3(&triangle->tangent_to_world.Y);

        crossVec3(&triangle->tangent_to_world.X,
                  &triangle->tangent_to_world.Y,
                  &triangle->tangent_to_world.Z);
        crossVec3(&triangle->tangent_to_world.Z,
                  &triangle->tangent_to_world.X,
                  &triangle->tangent_to_world.Y);

        norm3(&triangle->tangent_to_world.Z);
        norm3(&triangle->tangent_to_world.Y);

        transposeMat3(&triangle->tangent_to_world,
                      &triangle->world_to_tangent);
    }
}