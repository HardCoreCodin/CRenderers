#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"

vec3 tetrahedron_initial_vertex_positions[4] = {
        {0, 0, 0},
        {0, 1, 1},
        {1, 1, 0},
        {1, 0, 1},
};

void updateTetrahedronPosition(Tetrahedron *tet, vec3 *position) {
    vec3 movement;
    subVec3(position, &tet->xform.position, &movement);
    for (u8 i = 0; i < 4; i++) iaddVec3(&tet->vertices[i], &movement);
    tet->xform.position = *position;
}

void updateTetrahedronMatrices(Tetrahedron *tet) {
    mat3 scale, skew, transform;
    setMat3ToIdentity(&skew);
    setMat3ToIdentity(&scale);
    scale.X.x = 1 / (tet->radius * 2*SQRT2/SQRT3);
    scale.Y.y = 1 / (tet->radius * SQRT2);
    skew.Y.x = -0.5;
    mulMat3(&scale, &skew, &transform);

    Triangle *t = tet->triangles;
    for (u8 i = 0; i < 4; i++, t++) {
        transposeMat3(&t->tangent_to_world, &t->world_to_tangent);
        imulMat3(&t->world_to_tangent, &transform);
    }
}

void updateTetrahedronRadius(Tetrahedron *tet, f32 radius) {
    f32 factor = radius / tet->radius;
    for (u8 i = 0; i < 4; i++) {
        isubVec3(&tet->vertices[i], &tet->xform.position);
        iscaleVec3(&tet->vertices[i], factor);
        iaddVec3(&tet->vertices[i], &tet->xform.position);
    }
    tet->radius = radius;
    updateTetrahedronMatrices(tet);
}

void initTetrahedron(Tetrahedron *tet, IndexBuffers *index_buffers, f32 radius) {
    initXform3(&tet->xform);
    tet->radius = radius;

    // Init vertex positions:
    for (u8 i = 0; i < 4; i++) tet->vertices[i] = tetrahedron_initial_vertex_positions[i];

    // Init triangles:
    Triangle *t = tet->triangles;
    for (u8 i = 0; i < 4; i++, t++) {
        t->v1 = index_buffers->tetrahedron[i][0];
        t->v2 = index_buffers->tetrahedron[i][1];
        t->v3 = index_buffers->tetrahedron[i][2];

        subVec3(&tet->vertices[t->v3],
                &tet->vertices[t->v1],
                &t->tangent_to_world.X);
        subVec3(&tet->vertices[t->v2],
                &tet->vertices[t->v1],
                &t->tangent_to_world.Y);

        crossVec3(&t->tangent_to_world.X,
                  &t->tangent_to_world.Y,
                  &t->tangent_to_world.Z);
        crossVec3(&t->tangent_to_world.Z,
                  &t->tangent_to_world.X,
                  &t->tangent_to_world.Y);

        norm3(&t->tangent_to_world.X);
        norm3(&t->tangent_to_world.Y);
        norm3(&t->tangent_to_world.Z);

        invertVec3(&t->tangent_to_world.Z);
    }

    f32 half_cube_edge  = tet->radius / SQRT3;
    vec3 offset;
    fillVec3(&offset, half_cube_edge);

    vec3 *vertex_poisition = tet->vertices;
    for (u8 i = 0; i < 4; i++, vertex_poisition++) {
        iscaleVec3(vertex_poisition, half_cube_edge + half_cube_edge);
        isubVec3(vertex_poisition, &offset);
    }

    updateTetrahedronMatrices(tet);
}