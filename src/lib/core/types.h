#pragma once
#pragma warning(disable : 4201)

#ifndef __cplusplus
#define false 0
#define true 1
#define EPS 0.0001f
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
    vec2 uv;
    vec3 position,
         normal,
         ray_origin,
         ray_direction;
    f32 distance,
         n1_over_n2,
         n2_over_n1;
    u8 hit_depth;
    bool inner;
} RayHit;

typedef struct {
    f32 RLdotL,
        NdotL,
        NdotV,
        NdotH;
    vec3 *N, *V, L, H, RL, RV;

    bool has_NdotL,
         has_NdotV,
         has_NdotH,
         has_RLdotL,
         has_H,
         has_RL,
         had_RR;
} HitInfo;

typedef f32 (*DiffuseShader)(HitInfo *hit_info, f32 intensity);
typedef f32 (*SpecularShader)(HitInfo *hit_info, f32 intensity, u8 exponent);
typedef bool (*ReflectionShader)(RayHit *hit, HitInfo *hit_info, vec3* reflected_color);
typedef bool (*RefractionShader)(RayHit *hit, HitInfo *hit_info, vec3* refracted_color);

typedef struct {
    vec3 diffuse_color;
    f32 specular_intensity, diffuse_intensity;
    u8 specular_exponent, uses;
    bool has_diffuse, has_specular, has_reflection, has_refraction, has_transparency;
    DiffuseShader diffuse_shader;
    SpecularShader specular_shader;
    ReflectionShader reflection_shader;
    RefractionShader refraction_shader;
} Material;

typedef struct {
    Material* material;
    vec3 position,
         normal;
} Plane;

typedef struct {
    Material* material;
    vec3 position;
    Bounds2Di bounds;
    f32 radius;
    bool in_view;
} Sphere;

typedef struct {
    vec3 color;
    vec3 position;
    f32 intensity;
} PointLight;

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
    PointLight *point_lights;
    Material *materials;
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