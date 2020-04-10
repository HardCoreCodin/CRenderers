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
typedef struct { s16 x, y;    } Coords2;
typedef struct { f32 x, y;    } Vector2;
typedef struct { f32 x, y, z; } Vector3;
typedef struct { u8  B, G, R; } Color;
typedef struct { f32 radius; Vector3 *world_position, *view_position; } Sphere;
typedef struct { u8 sphere_count; Sphere *spheres; } Scene;
typedef struct { f32 distance; Vector3 position, normal; } RayHit;
typedef struct { f32 zoom, dolly, orbit, orient, pan; } Sensitivity;
typedef struct { u32 key; bool is_pressed; } Button;
typedef struct { Button forward, back, left, right, up, down, hud, first, second; } Buttons;
typedef struct { Coords2 absolute, relative; } MouseCoords;
typedef struct { Vector2 coords; u64 ticks; } MouseButtonState;
typedef struct { MouseButtonState up, down; bool is_down, clicked; } MouseButton;
typedef struct { MouseButton left, right, middle; } MouseButtons;
typedef struct { f32 scroll_amount; bool was_scrolled; } MouseWheel;
typedef struct {
    bool is_captured, has_moved, double_clicked;
    MouseWheel wheel;
    MouseCoords coords;
    MouseButtons buttons;
} Mouse;


typedef union {
    struct {
        f32 m11, m12,
            m21, m22;
    };
    struct {
        Vector2 i, j;
    };
} Matrix2x2;

typedef union {
    struct {
        f32 m11, m12, m13,
            m21, m22, m23,
            m31, m32, m33;
    };
    struct {
        Vector3 i, j, k;
    };
} Matrix3x3;

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
    Matrix3x3 *yaw, *pitch, *roll, *rotation, *rotation_inverted;
    Vector3 *position, *forward, *right, *up;
} Transform3D;

typedef struct {
    f32 focal_length;
    Transform3D *transform;
} Camera3D;

typedef struct Camera2D {
    f32 focal_length;
    Transform2D* transform;
} Camera2D;

typedef struct { Callback mouse_moved, mouse_scrolled, update;} ControllerCallbacks;
typedef struct {
    bool moved, rotated, zoomed;
    f32 mouse_movement_speed, mouse_scroll_speed;
    ControllerCallbacks on;
    Vector3* movement;
} Controller;

typedef struct { Callback resized, render, double_clicked; } RendererCallbacks;
typedef struct {
    char* title;
    RendererCallbacks on;
    Controller* controller;
} Renderer;

typedef struct {
    bool is_running, in_fps_mode;
    Renderer* renderer;
    Callback updateAndRender;
} Engine;