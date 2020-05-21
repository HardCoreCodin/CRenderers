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

typedef struct Engine Engine;
typedef void (*EngineCallback)(Engine* engine);
typedef void (*UpdateWindowTitle)();
typedef void (*PrintDebugString)(char* str);
typedef u64 (*GetTicks)();
typedef struct PerfTicks {
    u64 before, after;
} PerfTicks;

typedef struct PerfDelta {
    u64 ticks;
    f64 seconds;
} PerfDelta;

typedef struct PerfAccum {
    u64 ticks, frames;
} PerfAccum;

typedef struct PerfAvg {
    f64
    frames_per_tick,
    ticks_per_frame;

    u16
    frames_per_second,
    milliseconds_per_frame,
    microseconds_per_frame,
    nanoseconds_per_frame;
} PerfAvg;

typedef struct Perf {
    GetTicks getTicks;

    PerfDelta delta;
    PerfAccum accum;
    PerfTicks ticks;
    PerfAvg avg;

    u64
    ticks_per_interval,
    ticks_per_second;

    f64
    seconds_per_tick,
    milliseconds_per_tick,
    microseconds_per_tick,
    nanoseconds_per_tick;
} Perf;

#define HUD_LENGTH 100
typedef struct HUD {
    bool is_visible;
    char text[HUD_LENGTH];
    char* width;
    char* height;
    char* fps;
    char* msf;
    char* mode;
    char* rat;
    char* perf;

    Perf* debug_perf;
} HUD;

typedef struct {
    s16 x, y;
    bool changed;
} Coords2;

typedef struct {
    f32 x, y;
} Vector2;

typedef struct {
    f32 x, y, z;
} Vector3;

typedef struct {
    Vector2
    *x_axis,
    *y_axis;
} Matrix2x2;

typedef struct {
    Vector3
    *x_axis,
    *y_axis,
    *z_axis;
} Matrix3x3;

typedef struct {
    f32 radius;

    Vector3
    *world_position,
    *view_position;
} Sphere;

typedef struct {
    f32 distance;

    Vector3
    position,
    normal;
} RayHit;

typedef struct {
    u32 key;
    bool is_pressed;
} Button;

typedef struct {
    Button
    forward,
    back,
    left,
    right,
    up,
    down,
    hud,
    first,
    second;
} Keyboard;

typedef struct {
    Coords2 absolute, relative;
} MouseCoords;

typedef struct {
    Vector2 coords;
    u64 ticks;
} MouseButtonState;

typedef struct {
    MouseButtonState up, down;
    bool is_down, clicked;
} MouseButton;

typedef struct {
    MouseButton left, right, middle;
} MouseButtons;

typedef struct {
    f32 scroll;
    bool changed;
} MouseWheel;

typedef struct {
    bool is_captured, double_clicked;
    MouseWheel wheel;
    MouseCoords coords;
    MouseButtons buttons;
} Mouse;

typedef struct {
    Matrix2x2* rotation;

    Vector2
    *forward,
    *position,
    *right;
} Transform2D;

typedef struct {
    Matrix3x3
    *yaw,
    *pitch,
    *roll,
    *rotation;

    Vector3
    *position,
    *forward,
    *right,
    *up;
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
typedef struct {
    bool
    position,
    orientation,
    fov;
} ControllerChanged;

typedef struct { EngineCallback
    mouseMoved,
    mouseScrolled,
    update;
} ControllerCallbacks;

typedef struct {
    enum ControllerType type;
    ControllerChanged changed;
    Camera* camera;
    ControllerCallbacks on;
} Controller;

typedef struct { Controller controller;
    f32
    max_velocity,
    max_acceleration,
    orientation_speed,
    zoom_speed,
    delta_time;

    Vector3
    *current_velocity,
    *target_velocity,
    *movement;
} FpsController;

typedef struct {Controller controller;
    f32
    pan_speed,
    dolly_speed,
    orbit_speed,
    dolly_amount,
    dolly_ratio,
    target_distance;

    Vector3
    *target_position,
    *movement,
    *scaled_right,
    *scaled_up;
} OrbController;

typedef struct {
    FpsController* fps;
    OrbController* orb;
} Controllers;

typedef struct {
    Camera *camera;
    u8 sphere_count;
    Sphere *spheres;
} Scene;

typedef struct { EngineCallback
    zoom,
    move,
    rotate,
    resize,
    render;
} RendererCallbacks;

enum RendererType {
    RENDERER_RT,
    RENDERER_RC
};
typedef struct {
    enum RendererType type;
    char* title;
    RendererCallbacks on;
} Renderer;

typedef struct {
    Renderer renderer;
    u32 ray_count;
    u8 rays_per_pixel;
    RayHit* closest_hit;
    Vector3 *ray_directions;
    Matrix3x3* inverted_camera_rotation;
} RayTracer;

typedef struct RayCaster {
    Renderer renderer;
    u32 ray_count;
    u8 rays_per_pixel;
    Vector2* ray_directions;
} RayCaster;

typedef struct {
    u8  B, G, R, A;
} Color;

typedef union {
    Color color;
    u32 value;
} Pixel;

typedef struct FrameBuffer {
    u16 width, height;
    u32 size;
    Pixel* pixels;
} FrameBuffer;

typedef struct {
    RayTracer* ray_tracer;
    RayCaster* ray_caster;
} Renderers;

typedef struct {
    Renderer* renderer;
    Controller* controller;
} Viewport;


typedef struct Engine {
    UpdateWindowTitle updateWindowTitle;
    PrintDebugString printDebugString;

    Controllers controllers;
    Renderers renderers;

    FrameBuffer* frame_buffer;
    Viewport* active_viewport;

    Scene* scene;

    Perf* perf;
    HUD* hud;

    Mouse* mouse;
    Keyboard* keyboard;

    bool is_running;
} Engine;
