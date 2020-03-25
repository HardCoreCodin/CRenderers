#include <math.h>

#include "lib/core/types.h"
#include "lib/controls/fps.h"
#include "lib/controls/orbit.h"
#include "lib/controls/controls.h"
#include "lib/core/memory.h"
#include "lib/render/buffers.h"
#include "lib/math/math2D.h"
#include "lib/core/engine3D.h"

const float COLOR_FACTOR = 0xFF / 2.0f;
static char* TITLE = "RayTrace";

typedef struct Sphere {
    f32 radius;
    Vector3 position;
} Sphere;

typedef struct Ray {
    Vector3* origin;
    Vector3* direction;
} Ray;

typedef struct RayHit {
    f32 distance,
        delta_of_squares,
        origin_to_closest,
        origin_to_closest_minus_delta_of_squares;
    Vector3* normal;
    Vector3* position;
} RayHit;

typedef struct RenderEngine {
    EngineCore3D* core;

    Ray ray;
    RayHit current_hit, closest_hit;

    u32 ray_count;
    u8 rays_per_pixel, sphere_count;
    Sphere* spheres;

    Vector3* source_ray_directions;
    Vector3* ray_directions;
    Vector3* ray_origin_to_sphere_center;
    Vector3* ray_origin_to_closest_position;
    Vector3* closest_position_to_sphere_center;

    Transform3D* transform;
} RenderEngine;

static RenderEngine render_engine;

void rotateRayDirections() {
    Matrix3x3* rotation_matrix = render_engine.core->camera.transform->rotation;
    Vector3* source_ray_direction = render_engine.source_ray_directions;
    Vector3* ray_direction = render_engine.ray_directions;

    for (u32 ray_index = 0; ray_index < render_engine.ray_count; ray_index++)
        mul3D(source_ray_direction++, rotation_matrix, ray_direction++);
}

void generateRayDirectionsRat() {
    Vector3* ray_direction = render_engine.source_ray_directions;

    f32 projection_plane_half_width = 1 / render_engine.core->camera.focal_length;

    f32 w = frame_buffer.width;
    f32 h = frame_buffer.height;
    f32 h_over_w = h / w;

    f32 x_step = -projection_plane_half_width / w;
    f32 y_step = x_step * h_over_w;

    Matrix3x3 yaw, pitch;
    setMatrix3x3ToIdentity(&yaw);
    setMatrix3x3ToIdentity(&pitch);

    setYaw3D(-x_step, &yaw);
    setPitch3D(-y_step, &pitch);

    Vector3 starting_ray_direction = {0, 0, 1};
    for (int i = 0; i < frame_buffer.height / 2; i++) imul3D(&starting_ray_direction, &pitch);
    for (int i = 0; i < frame_buffer.width / 2; i++) imul3D(&starting_ray_direction, &yaw);


//
//    setYaw3D(projection_plane_half_width / 2 + x_step / 2, &yaw);
//
//    setPitch3D(h_over_w * projection_plane_half_width / 2 + y_step / 2, &pitch);
//
//    rotate3D(projection_plane_half_width / 2 + x_step / 2,
//             h_over_w * projection_plane_half_width / 2 + y_step / 2, 0,
//             render_engine.transform);
//    Vector3 starting_ray_direction = render_engine.transform->rotation->k;
//
//    imul3D(&starting_ray_direction, &pitch);
//    imul3D(&starting_ray_direction, &yaw);

    setYaw3D(x_step, &yaw);
    setPitch3D(y_step, &pitch);

    Vector3 current_direction;
    //
//    setMatrix3x3ToIdentity(render_engine.transform->yaw);
//    setMatrix3x3ToIdentity(render_engine.transform->pitch);
//    rotate3D(x_step, y_step, 0, render_engine.transform);

    for (u16 pixel_y = 0; pixel_y < frame_buffer.height; pixel_y++) {
        current_direction.x = starting_ray_direction.x;
        current_direction.y = starting_ray_direction.y;
        current_direction.z = starting_ray_direction.z;

        for (u16 pixel_x = 0; pixel_x < frame_buffer.width; pixel_x++) {
//            imul3D(ray_direction, render_engine.transform->yaw);
            ray_direction->x = current_direction.x;
            ray_direction->y = current_direction.y;
            ray_direction->z = current_direction.z;
            ray_direction++;

            imul3D(&current_direction, &yaw);
        }

//        imul3D(&starting_ray_direction, render_engine.transform->pitch);
        imul3D(&starting_ray_direction, &pitch);
    }

//    f32 x_step = projection_plane_half_width / w;
//    f32 y_step = -x_step * h_over_w;
//
//    f32 x_start = -projection_plane_half_width / 2 + x_step / 2;
//    f32 y_start = (projection_plane_half_width / 2) * h_over_w  + y_step / 2;
//
//    f32 t_x, t_y = y_start;
//
//    for (u16 pixel_y = 0; pixel_y < frame_buffer.height; pixel_y++) {
//        t_x = x_start;
//
//        for (u16 pixel_x = 0; pixel_x < frame_buffer.width; pixel_x++) {
//            f32 z = -2 * (t_x + t_y) / (t_x * t_x + t_y * t_y + 1);
//            f32 x = t_x * z + 1;
//            f32 y = t_y * z + 1;
//
//            ray_direction->x = x * 0.5f;
//            ray_direction->y = y * 0.5f;
//            ray_direction->z = z * 0.5f;
////            setPointOnUnitSphere(t_x, t_y, ray_direction);
//            ray_direction++;
//
//            t_x += x_step;
//        }
//
//        t_y += y_step;
//    }
}


void generateRayDirections(u16 width, u16 height) {
    f32 focal_length = render_engine.core->camera.focal_length;
    Vector3* source_ray_direction = render_engine.source_ray_directions;

    f32 ray_direction_length = 0;
    f32 squared_focal_length = focal_length * focal_length;
    f32 one_over_width = 1.0f / (f32)width;
    f32 x_dir = 0, x_dir_squared = 0;
    f32 y_dir = 0, y_dir_squared= 0;

    for (u16 pixel_y = height; pixel_y > 0; pixel_y--) {
        y_dir = (((f32)pixel_y - 0.5f) * one_over_width) - 0.5f;
        y_dir_squared = y_dir * y_dir;

        for (u16 pixel_x = 0; pixel_x < width; pixel_x++) {
            x_dir = (((f32)pixel_x + 0.5f) * one_over_width) - 0.5f;
            x_dir_squared = x_dir * x_dir;

            ray_direction_length = sqrtf(x_dir_squared + y_dir_squared + squared_focal_length);

            source_ray_direction->x = x_dir / ray_direction_length;
            source_ray_direction->y = y_dir / ray_direction_length;
            source_ray_direction->z = focal_length / ray_direction_length;
            source_ray_direction++;
        }
    }
}

u8 rayIntersectsASphere() {
    f32 squared_radius = 0;
    f32 squared_distance_from_closest_position_to_center = 0;
    render_engine.closest_hit.origin_to_closest_minus_delta_of_squares = 10000;


    Sphere* sphere = render_engine.spheres;
    Sphere* closest_sphere = 0;

    for (u32 sphere_index = 0; sphere_index < render_engine.sphere_count; sphere_index++) {
        squared_radius = sphere->radius * sphere->radius;

        sub3D(&sphere->position, render_engine.ray.origin, render_engine.ray_origin_to_sphere_center);

        render_engine.current_hit.origin_to_closest = dot3D(render_engine.ray_origin_to_sphere_center, render_engine.ray.direction);
        if (render_engine.current_hit.origin_to_closest > 0) {

            scale3D(render_engine.ray.direction, render_engine.current_hit.origin_to_closest, render_engine.ray_origin_to_closest_position);
            add3D(render_engine.ray.origin, render_engine.ray_origin_to_closest_position, render_engine.current_hit.position);
            sub3D(&sphere->position, render_engine.current_hit.position, render_engine.closest_position_to_sphere_center);
            
            squared_distance_from_closest_position_to_center = squaredLength3D(render_engine.closest_position_to_sphere_center);
            if (squared_distance_from_closest_position_to_center <= squared_radius) {

                render_engine.current_hit.delta_of_squares = squared_radius - squared_distance_from_closest_position_to_center;
                render_engine.current_hit.origin_to_closest_minus_delta_of_squares = render_engine.current_hit.origin_to_closest - render_engine.current_hit.delta_of_squares;
                if (render_engine.current_hit.origin_to_closest_minus_delta_of_squares > 0 &&
                    render_engine.current_hit.origin_to_closest_minus_delta_of_squares <= render_engine.closest_hit.origin_to_closest_minus_delta_of_squares) {
                    render_engine.closest_hit = render_engine.current_hit;
                    closest_sphere = sphere;
                }
            }
        }

        sphere++;
    }

    if (closest_sphere) {
        if (render_engine.closest_hit.delta_of_squares > 0.001f) {
            render_engine.closest_hit.distance = render_engine.closest_hit.origin_to_closest - sqrtf(render_engine.closest_hit.delta_of_squares);
            scale3D(render_engine.ray.direction, render_engine.closest_hit.distance, render_engine.closest_hit.position);
            iadd3D(render_engine.closest_hit.position, render_engine.ray.origin);
        }

        sub3D(render_engine.closest_hit.position, &closest_sphere->position, render_engine.closest_hit.normal);
        if (closest_sphere->radius != 1)
            iscale3D(render_engine.closest_hit.normal, 1 / closest_sphere->radius);
    }

    return closest_sphere ? true : false;
}

void render() {
    u32* pixel = frame_buffer.pixels;
    render_engine.ray.direction = render_engine.ray_directions;
    Color color;

    for (u32 i = 0; i < frame_buffer.size; i++) {
        if (rayIntersectsASphere()) {
            color.R = (u8)((render_engine.closest_hit.normal->x + 1) * COLOR_FACTOR);
            color.G = (u8)((render_engine.closest_hit.normal->y + 1) * COLOR_FACTOR);
            color.B = (u8)((render_engine.closest_hit.normal->z + 1) * COLOR_FACTOR);
        } else color.value = 0;

        *pixel++ = color.value;
        render_engine.ray.direction++;
    }
}

void resetRayDirections(u16 width, u16 height) {
    generateRayDirections(width, height);
    rotateRayDirections();
}

void onFrameBufferResized(u16 width, u16 height) {
    resetRayDirections(width, height);
}

void onMousePositionChanged(f32 dx, f32 dy, Mouse* mouse, Buttons* buttons) {
    onMousePositionChanged3D(dx, dy, mouse, buttons, render_engine.core);
}

void onMouseWheelChanged(f32 amount, Mouse* mouse) {
    onMouseWheelChanged3D(amount, mouse, render_engine.core);
}

void update(f32 delta_time, u16 width, u16 height, Controls* controls) {
    if (delta_time > 1)
        delta_time = 1;

    Camera3D* camera = &render_engine.core->camera;
    FpsController3D* fps_controller = &render_engine.core->fps_controller;
    OrbitController3D* orbit_controller = &render_engine.core->orbit_controller;

    if (controls->mouse.is_captured) {

        if (fps_controller->zoom.changed) {
            zoom(camera, fps_controller);
            resetRayDirections(width, height);
        }

        if (fps_controller->orientation.changed) {
            look(camera, fps_controller);
            rotateRayDirections(
                    render_engine.source_ray_directions,
                    render_engine.ray_directions,
                    render_engine.ray_count,
                    camera->transform->rotation);
        }
    } else {
        if (render_engine.core->orbit_controller.orbit.changed) {
            orbit(camera, orbit_controller);
            rotateRayDirections(
                    render_engine.source_ray_directions,
                    render_engine.ray_directions,
                    render_engine.ray_count,
                    camera->transform->rotation);
        }

        if (orbit_controller->pan.changed)
            pan(camera, orbit_controller);

        if (orbit_controller->dolly.changed)
            dolly(camera, orbit_controller);
    }

    move(camera, fps_controller, &controls->keyboard, &controls->buttons, delta_time);
}


void initHit(RayHit* hit) {
    hit->normal = (Vector3*)allocate(sizeof(Vector3));
    hit->position = (Vector3*)allocate(sizeof(Vector3));

    fill3D(hit->normal, 0);
    fill3D(hit->position, 0);
    hit->distance = hit->delta_of_squares = hit->origin_to_closest = hit->origin_to_closest_minus_delta_of_squares = 0;
};

void initRenderEngine(u16 width, u16 height) {
    render_engine.core = (EngineCore3D*)allocate(sizeof(EngineCore3D));

    initEngineCore3D(render_engine.core);
    initHit(&render_engine.current_hit);
    initHit(&render_engine.closest_hit);

    render_engine.rays_per_pixel = 1;
    render_engine.ray_count = width * height * render_engine.rays_per_pixel;

    render_engine.core->camera.position->x = 5;
    render_engine.core->camera.position->y = 5;
    render_engine.core->camera.position->z = -10;

    u8 radius = 1;
    u8 horizontal_count = 2;
    u8 vertical_count = 2;

    render_engine.sphere_count = horizontal_count * vertical_count;
    render_engine.spheres = (Sphere*)allocate(sizeof(Sphere) * render_engine.sphere_count);

    Sphere* sphere = render_engine.spheres;

    u8 gap = radius * 3;
    u8 sphere_x = 0;
    u8 sphere_z = 0;

    for (u8 z = 0; z < vertical_count; z++) {
        sphere_x = 0;

        for (u8 x = 0; x < horizontal_count; x++) {
            sphere->radius = 1;

            sphere->position.x = sphere_x;
            sphere->position.y = 0;
            sphere->position.z = sphere_z;

            sphere_x += gap;
            sphere++;
        }

        sphere_z += gap;
    }

    render_engine.transform = (Transform3D*)allocate(sizeof(Transform3D));
    initTransform3D(render_engine.transform);

    render_engine.source_ray_directions = (Vector3*)allocate(sizeof(Vector3) * render_engine.ray_count);
    render_engine.ray_directions = (Vector3*)allocate(sizeof(Vector3) * render_engine.ray_count);

    render_engine.ray_origin_to_sphere_center = (Vector3*)allocate(sizeof(Vector3));
    render_engine.ray_origin_to_closest_position = (Vector3*)allocate(sizeof(Vector3));

    render_engine.closest_position_to_sphere_center = (Vector3*)allocate(sizeof(Vector3));

    render_engine.ray.origin = render_engine.core->camera.position;
}
