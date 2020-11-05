#pragma once

#include "lib/core/types.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"

void initXform2(xform2 *xform) {
    setMat2ToIdentity(&xform->matrix);
    setMat2ToIdentity(&xform->rotation_matrix);
    setMat2ToIdentity(&xform->rotation_matrix_inverted);
    fillVec2(&xform->position, 0);
    xform->forward_direction = &xform->rotation_matrix.X;
    xform->right_direction   = &xform->rotation_matrix.Y;
}

void initXform3(xform3 *xform) {
    setMat3ToIdentity(&xform->matrix);
    setMat3ToIdentity(&xform->yaw_matrix);
    setMat3ToIdentity(&xform->pitch_matrix);
    setMat3ToIdentity(&xform->roll_matrix);
    setMat3ToIdentity(&xform->rotation_matrix);
    setMat3ToIdentity(&xform->rotation_matrix_inverted);
    fillVec3(&xform->position, 0);
    xform->right_direction   = &xform->rotation_matrix.X;
    xform->up_direction      = &xform->rotation_matrix.Y;
    xform->forward_direction = &xform->rotation_matrix.Z;
}
//
//inline void rotate2D(xform2 *xform, f32 yaw, f32 pitch) {
//    if (!yaw) return;
//    yaw2(yaw, &xform->rotation_matrix);
//    transposeMat2(&xform->rotation_matrix, &xform->rotation_matrix_inverted);
//    mulMat2(&xform->matrix, &xform->rotation_matrix, &xform->matrix);
//}

inline void rotateXform3(xform3 *xform, f32 yaw, f32 pitch, f32 roll) {
    if (yaw)   yawMat3(  yaw,   &xform->yaw_matrix);
    if (pitch) pitchMat3(pitch, &xform->pitch_matrix);
    if (roll)  rollMat3( roll,  &xform->roll_matrix);
    mulMat3(&xform->pitch_matrix,
            &xform->yaw_matrix,
            &xform->rotation_matrix);
    imulMat3(&xform->rotation_matrix,
             &xform->roll_matrix);
    imulMat3(&xform->matrix,
             &xform->rotation_matrix);
    transposeMat3(&xform->rotation_matrix,
                  &xform->rotation_matrix_inverted);
}