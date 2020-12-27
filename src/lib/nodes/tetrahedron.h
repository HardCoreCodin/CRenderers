#pragma once

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/globals/scene.h"

void updateTetrahedronPosition(Tetrahedron *tet, vec3 *position) {
    vec3 movement;
    subVec3(position, &tet->node.position, &movement);
    for (u8 i = 0; i < 4; i++) iaddVec3(&tet->vertices[i], &movement);
    tet->node.position = *position;
}

void updateTetrahedronMatrices(Tetrahedron *tet) {
    mat3 scale, skew, transform;
    setMat3ToIdentity(&skew);
    setMat3ToIdentity(&scale);
    scale.X.x = 1 / (tet->node.radius * 2*SQRT2/SQRT3);
    scale.Y.y = 1 / (tet->node.radius * SQRT2);
    skew.Y.x = -0.5;
    mulMat3(&scale, &skew, &transform);

    for (u8 i = 0; i < 4; i++) {
        transposeMat3(&tet->tangent_to_world[i], &tet->world_to_tangent[i]);
        imulMat3(&tet->world_to_tangent[i], &transform);
    }
}

void updateTetrahedronRadius(Tetrahedron *tet, f32 radius) {
    f32 factor = radius / tet->node.radius;
    for (u8 i = 0; i < 4; i++) {
        isubVec3(&tet->vertices[i], &tet->node.position);
        iscaleVec3(&tet->vertices[i], factor);
        iaddVec3(&tet->vertices[i], &tet->node.position);
    }
    tet->node.radius = radius;
    updateTetrahedronMatrices(tet);
}

void initTetrahedron(Tetrahedron *tet, f32 radius, vec3 *vertices, TriangleIndices *indices) {
    tet->node.radius = radius;

    // Init vertex positions:
    for (u8 i = 0; i < 4; i++) tet->vertices[i] = vertices[i];

    // Init triangles:
    for (u8 i = 0; i < 4; i++) {
        subVec3(&tet->vertices[indices[i].v3],
                &tet->vertices[indices[i].v1],
                &tet->tangent_to_world[i].X);
        subVec3(&tet->vertices[indices[i].v2],
                &tet->vertices[indices[i].v1],
                &tet->tangent_to_world[i].Y);

        crossVec3(&tet->tangent_to_world[i].X,
                  &tet->tangent_to_world[i].Y,
                  &tet->tangent_to_world[i].Z);
        crossVec3(&tet->tangent_to_world[i].Z,
                  &tet->tangent_to_world[i].X,
                  &tet->tangent_to_world[i].Y);

        norm3(&tet->tangent_to_world[i].X);
        norm3(&tet->tangent_to_world[i].Y);
        norm3(&tet->tangent_to_world[i].Z);

        invertVec3(&tet->tangent_to_world[i].Z);
    }

    f32 half_cube_edge  = tet->node.radius / SQRT3;
    vec3 offset;
    fillVec3(&offset, half_cube_edge);

    vec3 *vertex_poisition = tet->vertices;
    for (u8 i = 0; i < 4; i++, vertex_poisition++) {
        iscaleVec3(vertex_poisition, half_cube_edge + half_cube_edge);
        isubVec3(vertex_poisition, &offset);
    }

    updateTetrahedronMatrices(tet);
}