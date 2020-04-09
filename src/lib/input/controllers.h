#pragma once

#include "lib/core/types.h"
#include "lib/input/keyboard.h"
#include "lib/math/math1D.h"
#include "lib/math/math2D.h"
#include "lib/math/math3D.h"
#include "lib/nodes/camera.h"
#include "lib/nodes/transform.h"
#include "lib/memory/allocators.h"

typedef struct Zoom {
    f32 sensitivity, in;
    bool changed;
} Zoom;

typedef struct Orientation {
    f32 sensitivity, yaw, pitch;
    bool changed;
} Orientation;

typedef struct Pan   {
    f32 sensitivity, right, up;
    bool changed;
} Pan;

typedef struct Dolly {
    f32 sensitivity, in, current, factor;
    bool changed;
} Dolly;

typedef struct Orbit {
    f32 sensitivity, azimuth, elevation;
    bool changed;
} Orbit;

typedef struct Controller {
    Zoom zoom;
    Orbit orbit;
    Dolly dolly;
    Pan pan;
    Orientation orientation;
} Controller;

Controller controller = {
        {1 / 10.f, 0, false},
        {1 / 1000.0f, 0, 0, false},
        {1, 0, -5, 5, false},
        {1 / 100.0f, 0, 0, false},
        {7 / 10000.0f, 0, 0, false}
};

void updateOrientation2D(Transform2D* transform) {
    rotate2D(controller.orientation.yaw, transform->rotation);
    controller.orientation.yaw = 0;
    controller.orientation.changed = false;
}
