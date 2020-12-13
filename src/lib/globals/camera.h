#pragma once

#include "lib/core/types.h"

#define ZOOM_SPEED 0.002f
#define MAX_VELOCITY 4.0f
#define MAX_ACCELERATION 20.0f
#define MOUSE_TURN_SPEED 0.0005f
#define KEYBOARD_TURN_SPEED 1.0f

#define PAN_SPEED 0.01f
#define DOLLY_SPEED 1.0f
#define ORBIT_SPEED 0.0005f
#define ORBIT_TARGET_DISTANCE 10.0f

#define NEAR_CLIPPING_PLANE_DISTANCE 0.1f
#define FAR_CLIPPING_PLANE_DISTANCE 1000.0f

typedef struct {
    f32 focal_length;
    xform3 transform;
    vec3 projection_scale;
} Camera;
Camera main_camera;

enum ControllerType {
    CONTROLLER_ORB,
    CONTROLLER_FPS
};

typedef struct {
    enum ControllerType type;

    bool moved,
         turned,
         zoomed;

    CallBack onUpdate,
             onMouseMoved,
             onMouseWheelScrolled;

    Camera *camera;
} CameraController;
CameraController *current_camera_controller;

typedef struct { CameraController controller;
    f32 zoom_amount;
    vec3 movement,
         old_position,
         target_velocity,
         current_velocity;
} FpsCameraController;
FpsCameraController fps_camera_controller;

typedef struct { CameraController controller;
    f32 dolly_amount,
        target_distance;

    vec3 movement,
         target_position,
         scaled_right,
         scaled_up;
} OrbCameraController;
OrbCameraController orb_camera_controller;