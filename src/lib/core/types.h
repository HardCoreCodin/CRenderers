#pragma once
#pragma warning(disable : 4201)

#ifndef __cplusplus
#define false 0
#define true 1
typedef unsigned char      bool;
#endif

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

typedef signed   char      s8;
typedef signed   short     s16;
typedef signed   int       s32;
typedef signed   long long s64;

typedef float  f32;
typedef double f64;

typedef void (*Callback)();
typedef struct { s16 x, y; bool changed; } Coords2;
typedef struct { f32 x, y;    } Vector2;
typedef struct { f32 x, y, z; } Vector3;
typedef struct { u8  B, G, R; } Color;
typedef struct { Vector2 *x_axis, *y_axis; } Matrix2x2;
typedef struct { Vector3 *x_axis, *y_axis, *z_axis; } Matrix3x3;
typedef struct { f32 radius; Vector3 *world_position, *view_position; } Sphere;
typedef struct { f32 distance; Vector3 position, normal; } RayHit;
typedef struct { u32 key; bool is_pressed; } Button;
typedef struct { Button forward, back, left, right, up, down, hud, first, second; } Buttons;
typedef struct { Coords2 absolute, relative; } MouseCoords;
typedef struct { Vector2 coords; u64 ticks; } MouseButtonState;
typedef struct { MouseButtonState up, down; bool is_down, clicked; } MouseButton;
typedef struct { MouseButton left, right, middle; } MouseButtons;
typedef struct { f32 scroll; bool changed; } MouseWheel;
typedef struct {
    bool is_captured, double_clicked;
    MouseWheel wheel;
    MouseCoords coords;
    MouseButtons buttons;
} Mouse;

typedef union {
    struct {
        Color color;
        u8 depth;
    };
    u32 value;
} Pixel;

typedef struct {
    Matrix2x2* rotation;
    Vector2 *forward, *position, *right;
} Transform2D;

typedef struct {
    Matrix3x3 *yaw, *pitch, *roll, *rotation;
    Vector3 *position, *forward, *right, *up;
} Transform3D;

typedef struct {
    f32 focal_length;
    Transform3D* transform;
    Transform2D* transform2D;
} Camera;

enum ControllerType {
    CONTROLLER_ORB,
    CONTROLLER_FPS
};
typedef struct { bool position, orientation, fov; } ControllerChanged;
typedef struct {
    enum ControllerType type;
    ControllerChanged changed;
    Camera* camera;
} Controller;

typedef struct {
    Camera *camera;
    u8 sphere_count;
    Sphere *spheres;
} Scene;

enum RendererType {
    RENDERER_RT,
    RENDERER_RC
};
typedef struct {
    enum RendererType type;
    char* title;
    Scene* scene;
} Renderer;

typedef struct {
    bool in_fps_mode;
    Renderer* renderer;
    Controller* controller;
} Viewport;

typedef struct {
    Scene scene;
    Viewport viewport;
    bool is_running;
} Engine;
