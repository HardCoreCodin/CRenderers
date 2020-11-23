#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"

void initTetrahedron(Tetrahedron *tet) {
    initXform3(&tet->xform);

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

    Triangle *triangle = &tet->triangles[0];
    for (u8 i = 0; i < 4; i++, triangle++) {
        subVec3(triangle->p3,
                triangle->p1,
                &triangle->tangent_to_world.X);
        subVec3(triangle->p2,
                triangle->p1,
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

        triangle->normal = &triangle->tangent_to_world.Z;

        transposeMat3(&triangle->tangent_to_world,
                      &triangle->world_to_tangent);
    }
}