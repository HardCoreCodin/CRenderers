#include "core.h"
#include "linalg.h"
#include "scene.h"
#include <math.h>

const float COLOR_FACTOR = 0xFF / 2.0f;
static char* TITLE = "RayTracer";

typedef struct Hit {
    Vector3* position;
    Vector3* surface_normal;
} Hit;

typedef struct Ray {
    Vector3* origin;
    Vector3* direction;
} Ray;

u8 any_hit;
Hit hit;
Ray ray;


Vector3* source_ray_directions;
Vector3* source_ray_direction;
Vector3* ray_directions;
Vector3* ray_direction;
Vector3* ray_origin_to_sphere_center;
Vector3* ray_origin_to_closest_position;
Vector3* closest_position_to_sphere_center;
Vector3* closest_position;

f32 x, y, z,
    squared_radius,
    squared_delta,
    squared_focal_length,
    squared_distance,
    squared_distance_from_closest_position_to_center,
    
    ray_direction_length,
    distance_to_intersection,
    distance_from_origin_to_closest_position,
    origin_to_closest_minus_squared_delta;

void rotate_ray_directions() {
    m11 = camera.matrix.m11; m21 = camera.matrix.m21; m31 = camera.matrix.m31;
    m12 = camera.matrix.m12; m22 = camera.matrix.m22; m32 = camera.matrix.m32;
    m13 = camera.matrix.m13; m23 = camera.matrix.m23; m33 = camera.matrix.m33;

    source_ray_direction = source_ray_directions;
    ray_direction = ray_directions;

    for (u32 ray_index = 0; ray_index < frame_buffer.size; ray_index++) {
        x = source_ray_direction->x;
        y = source_ray_direction->y;
        z = source_ray_direction->z;

        ray_direction->x = x*m11 + y*m21 + z*m31;
        ray_direction->y = x*m12 + y*m22 + z*m32;
        ray_direction->z = x*m13 + y*m23 + z*m33;

        source_ray_direction++;
        ray_direction++;
    }
}

void generate_ray_directions() {
    ray_direction = source_ray_directions;
    squared_focal_length = camera.focal_length * camera.focal_length;

    f32 one_over_width = 1.0f / frame_buffer.width;
    f32 x_dir, x_dir_squared;
    f32 y_dir, y_dir_squared;

    for (u16 pixel_y = 0; pixel_y < frame_buffer.height; pixel_y++) {
        y_dir = ((pixel_y + 0.5f) * one_over_width) - 0.5f;
        y_dir_squared = y_dir * y_dir;

        for (u16 pixel_x = 0; pixel_x < frame_buffer.width; pixel_x++) {
            x_dir = ((pixel_x + 0.5f) * one_over_width) - 0.5f;
            x_dir_squared = x_dir * x_dir;

            ray_direction_length = (f32)sqrt(x_dir_squared + y_dir_squared + squared_focal_length);

            ray_direction->x = x_dir / ray_direction_length;
            ray_direction->y = y_dir / ray_direction_length;
            ray_direction->z = camera.focal_length / ray_direction_length;
            ray_direction++;
        }
    }

    ray_directions = ray_direction;
    rotate_ray_directions();
}

u8 ray_intersects_sphere() {
    squared_delta = 0;
    distance_to_intersection = 0;
    distance_from_origin_to_closest_position = 0;
    squared_distance_from_closest_position_to_center = 0;
    origin_to_closest_minus_squared_delta = 0;
    squared_radius = sphere.radius * sphere.radius;

    sub(sphere.position, ray.origin, ray_origin_to_sphere_center);
    distance_from_origin_to_closest_position = dot(ray_origin_to_sphere_center, ray.direction);
    if (distance_from_origin_to_closest_position <= 0)
        return 0;

    scale(ray.direction, distance_from_origin_to_closest_position, ray_origin_to_closest_position);
    add(ray.origin, ray_origin_to_closest_position, closest_position);
    sub(sphere.position, closest_position, closest_position_to_sphere_center);
    squared_distance_from_closest_position_to_center = length_squared(closest_position_to_sphere_center);
    if (squared_distance_from_closest_position_to_center > squared_radius)
        return 0;

    squared_delta = squared_radius - squared_distance_from_closest_position_to_center;
    origin_to_closest_minus_squared_delta = distance_from_origin_to_closest_position - squared_delta;
    if (origin_to_closest_minus_squared_delta > squared_distance)
        return 0;

    hit.position = closest_position;

    if (squared_delta > 0.001f) {
        distance_to_intersection = distance_from_origin_to_closest_position - (f32)sqrt(squared_delta);
        if (distance_to_intersection <= 0)
            return 0;

        scale(ray.direction, distance_to_intersection, hit.position);
        iadd(hit.position, ray.origin);
    }

    sub(hit.position, sphere.position, hit.surface_normal);
    if (sphere.radius != 1)
        idiv(hit.surface_normal, sphere.radius);

    squared_distance = origin_to_closest_minus_squared_delta;

    return 1;
}

void init_renderer() {
    source_ray_directions = (Vector3*)allocate_memory(RENDER_SIZE);
    ray_directions = (Vector3*)allocate_memory(RENDER_SIZE);
    
    ray_origin_to_sphere_center = (Vector3*)allocate_memory(sizeof(Vector3));
    ray_origin_to_closest_position = (Vector3*)allocate_memory(sizeof(Vector3));
    
    closest_position = (Vector3*)allocate_memory(sizeof(Vector3));
    closest_position_to_sphere_center = (Vector3*)allocate_memory(sizeof(Vector3));
    
    hit.position = (Vector3*)allocate_memory(sizeof(Vector3));
    hit.surface_normal = (Vector3*)allocate_memory(sizeof(Vector3));
    
    camera.position = ray.origin = (Vector3*)allocate_memory(sizeof(Vector3));
    camera.position->x = 5;
    camera.position->y = 5;
    camera.position->z = -10;
}

void render() {
    pixel = frame_buffer.pixels;
    ray.direction = ray_directions;

    for (u32 ray_index = 0; ray_index < frame_buffer.size; ray_index++) {
        squared_distance = 10000;
        any_hit = 0;

        sphere.position = sphere_positions;
        for (u32 sphere_index = 0; sphere_index < sphere_count; sphere_index++) {
            if (ray_intersects_sphere())
                any_hit = 1;

            sphere.position++;
        }

        if (any_hit) {
            pixel->R = (u8)((hit.surface_normal->x + 1) * COLOR_FACTOR);
            pixel->G = (u8)((hit.surface_normal->y + 1) * COLOR_FACTOR);
            pixel->B = (u8)((hit.surface_normal->z + 1) * COLOR_FACTOR);
        } else pixel->R = pixel->G = pixel->B = 0;
        pixel->A = 255;

        pixel++;
        ray.direction++;
    }
}

void on_resize() {
    generate_ray_directions();
}

void on_mouse_wheel(float amount) {
    camera.focal_length += amount / 10.0f;
    generate_ray_directions();
}

void on_mouse_move() {
    rotate_camera();
    rotate_ray_directions();
}

//void render() {
    //current_pixel = pixels;
    //for (int y = 0; y < height; ++y)
    //    for (int x = 0; x < width; ++x)
    //        *current_pixel++ = ((y << 16) | x);
//}