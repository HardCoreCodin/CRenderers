#pragma once
#pragma warning(disable : 4201)

#ifndef __cplusplus
#define false 0
#define true 1
#define EPS 0.000001f
typedef unsigned char      bool;
#endif

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

typedef signed   short     i16;
typedef signed   int       i32;

typedef float  f32;
typedef double f64;

typedef unsigned char byte;

typedef void (*UpdateWindowTitle)();
typedef void (*PrintDebugString)(char* str);
typedef u64 (*GetTicks)();
typedef void (*CallBack)();

typedef struct {
    f32 delta_time;
    u64 ticks_before,
        ticks_after,
        ticks_diff,
        accumulated_ticks,
        accumulated_frame_count,
        ticks_of_last_report,
        seconds,
        milliseconds,
        microseconds,
        nanoseconds;

    f64 average_frames_per_tick,
        average_ticks_per_frame;
    u16 average_frames_per_second,
        average_milliseconds_per_frame,
        average_microseconds_per_frame,
        average_nanoseconds_per_frame;
} Timer;

#define HUD_LENGTH 120
typedef struct HUD {
    bool is_visible;
    char text[HUD_LENGTH];
    char *width,
         *height,
         *fps,
         *msf,
         *spr,
         *pixels,
         *shading,
         *mode;
} HUD;

typedef struct { u8 B, G, R, A; } Color;

typedef struct { i32 x, y;    } vec2i;
typedef struct { f32 x, y;    } vec2;
typedef struct { f32 x, y, z; } vec3;

typedef struct { vec2 X, Y;    } mat2;
typedef struct { vec3 X, Y, Z; } mat3;

typedef struct { u16 min, max; } range2i;
typedef struct { range2i x_range, y_range; } Bounds2Di;

typedef struct {
    vec3 *world_position,
         *view_position,
         *world_normal,
         *view_normal;

    bool in_view;
} Plane;

typedef struct {
    f32 radius;
    vec3 *world_position,
         *view_position;
    Bounds2Di view_bounds;
    bool in_view;
} Sphere;

typedef struct {
    vec3 *world_position,
         *view_position;
    f32 intensity;
    Color color;
} Light;

typedef struct {
    f32 distance;

    vec3 position,
         normal;
    vec3 *ray_direction;
} RayHit;

typedef struct {
    bool is_pressed,
         is_released;

    vec2i down_pos,
          up_pos;
} MouseButton;

typedef struct {
    mat2 matrix,
         rotation_matrix,
         rotation_matrix_inverted;
    vec2 position,
         *right_direction,
         *forward_direction;
} xform2;

typedef struct {
    mat3 matrix,
         yaw_matrix,
         pitch_matrix,
         roll_matrix,
         rotation_matrix,
         rotation_matrix_inverted;

    vec3 position,
         *up_direction,
         *right_direction,
         *forward_direction;
} xform3;

typedef struct {
    f32 focal_length, one_over_focal_length;
    xform3 transform;
} Camera;

enum ControllerType {
    CONTROLLER_ORB,
    CONTROLLER_FPS
};
enum ShadingMode {
    Normal,
    Lambert,
    Phong,
    Blinn
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

typedef struct { CameraController controller;
    f32 zoom_amount;
    vec3 movement,
         old_position,
         target_velocity,
         current_velocity;
} FpsCameraController;

typedef struct { CameraController controller;
    f32 dolly_amount,
        target_distance;

    vec3 movement,
         target_position,
         scaled_right,
         scaled_up;
} OrbCameraController;

typedef struct {
    Camera *camera;
    Sphere *spheres;
    Plane *planes;
    Light *lights;
    u8 sphere_count,
       plane_count,
       light_count,
       active_sphere_count;
} Scene;

typedef struct {
    u32 ray_count;
    u8 rays_per_pixel;
    RayHit closest_hit;
    vec3 *ray_directions;
    mat3 inverted_camera_rotation;
} RayTracer;


typedef union {
    Color color;
    u32 value;
} Pixel;

typedef struct FrameBuffer {
    u16 width, height;
    u32 size, active_pixel_count;
    f32 height_over_width,
        width_over_height;
    Pixel* pixels;
} FrameBuffer;