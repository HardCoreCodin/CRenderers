#include <math.h>
#include "core.h"
#include "math3D.h"
#include "core3D.h"

const float COLOR_FACTOR = 0xFF / 2.0f;
static char* TITLE = "RayTrace";

typedef struct Sphere {
    Vector3* position; 
    f32 radius;
} Sphere;

static Sphere sphere;

Vector3* sphere_positions;
u8 sphere_count;

typedef struct Ray {
    Vector3* origin;
    Vector3* direction;
} Ray;

typedef struct Hit {
    f32 distance,
        delta_of_squares,
        origin_to_closest,
        origin_to_closest_minus_delta_of_squares;
    Vector3 position;
} Hit;

Hit current_hit = {0, 0, 0, 0};
Hit closest_hit = {0, 0, 0, 0};
Ray ray;

Vector3* source_ray_directions;
Vector3* source_ray_direction;
Vector3* ray_directions;
Vector3* ray_direction;
Vector3* ray_origin_to_sphere_center;
Vector3* ray_origin_to_closest_position;
Vector3* closest_position_to_sphere_center;
Vector3* closest_hit_surface_normal;
Vector3* closest_hit_sphere_position;


void rotateRayDirections() {
    source_ray_direction = source_ray_directions;
    ray_direction = ray_directions;

    for (u32 i = 0; i < frame_buffer.size; i++)
        mul3D(source_ray_direction++, &rotation_matrix, ray_direction++);
}

void generateRayDirections() {
    ray_direction = source_ray_directions;

    f32 ray_direction_length = 0;
    f32 squared_focal_length = camera.focal_length * camera.focal_length;
    f32 one_over_width = 1.0f / (f32)frame_buffer.width;
    f32 x_dir = 0, x_dir_squared = 0;
    f32 y_dir = 0, y_dir_squared= 0;

    for (u16 pixel_y = 0; pixel_y < frame_buffer.height; pixel_y++) {
        y_dir = (((f32)pixel_y + 0.5f) * one_over_width) - 0.5f;
        y_dir_squared = y_dir * y_dir;

        for (u16 pixel_x = 0; pixel_x < frame_buffer.width; pixel_x++) {
            x_dir = (((f32)pixel_x + 0.5f) * one_over_width) - 0.5f;
            x_dir_squared = x_dir * x_dir;

            ray_direction_length = sqrtf(x_dir_squared + y_dir_squared + squared_focal_length);

            ray_direction->x = x_dir / ray_direction_length;
            ray_direction->y = y_dir / ray_direction_length;
            ray_direction->z = camera.focal_length / ray_direction_length;
            ray_direction++;
        }
    }

    ray_directions = ray_direction;
}

u8 rayIntersectsASphere() {
    u8 any_hit = 0;
    f32 squared_radius = 0;
    f32 squared_distance_from_closest_position_to_center = 0;
    closest_hit.origin_to_closest_minus_delta_of_squares = 10000;

    sphere.position = sphere_positions;
    for (u32 sphere_index = 0; sphere_index < sphere_count; sphere_index++) {
        squared_radius = sphere.radius * sphere.radius;

        sub3D(sphere.position, ray.origin, ray_origin_to_sphere_center);

        current_hit.origin_to_closest = dot3D(ray_origin_to_sphere_center, ray.direction);
        if (current_hit.origin_to_closest > 0) {

            scale3D(ray.direction, current_hit.origin_to_closest, ray_origin_to_closest_position);
            add3D(ray.origin, ray_origin_to_closest_position, &current_hit.position);
            sub3D(sphere.position, &current_hit.position, closest_position_to_sphere_center);
            
            squared_distance_from_closest_position_to_center = squaredLength3D(closest_position_to_sphere_center);
            if (squared_distance_from_closest_position_to_center <= squared_radius) {

                current_hit.delta_of_squares = squared_radius - squared_distance_from_closest_position_to_center;
                current_hit.origin_to_closest_minus_delta_of_squares = current_hit.origin_to_closest - current_hit.delta_of_squares;
                if (current_hit.origin_to_closest_minus_delta_of_squares > 0 && 
                    current_hit.origin_to_closest_minus_delta_of_squares <= closest_hit.origin_to_closest_minus_delta_of_squares) {
                    closest_hit_sphere_position = sphere.position;
                    closest_hit = current_hit;
                    any_hit = 1;
                }
            }
        }

        sphere.position++;
    }

    if (any_hit) {
        if (closest_hit.delta_of_squares > 0.001f) {
            closest_hit.distance = closest_hit.origin_to_closest - sqrtf(closest_hit.delta_of_squares);
            scale3D(ray.direction, closest_hit.distance, &closest_hit.position);
            iadd3D(&closest_hit.position, ray.origin);
        }

        sub3D(&closest_hit.position, closest_hit_sphere_position, closest_hit_surface_normal);
        if (sphere.radius != 1)
            idiv3D(closest_hit_surface_normal, sphere.radius);
    }

    return any_hit;
}

void render() {
    pixel = frame_buffer.pixels;
    ray.direction = ray_directions;

    for (u32 i = 0; i < frame_buffer.size; i++) {
        if (rayIntersectsASphere()) {
            pixel->R = (u8)((closest_hit_surface_normal->x + 1) * COLOR_FACTOR);
            pixel->G = (u8)((closest_hit_surface_normal->y + 1) * COLOR_FACTOR);
            pixel->B = (u8)((closest_hit_surface_normal->z + 1) * COLOR_FACTOR);
            pixel->A = 255;
        } else pixel->color = 0;

        pixel++;
        ray.direction++;
    }
}
void onFrameBufferResized() {
    generateRayDirections();
    rotateRayDirections();
}

void update(f32 delta_time) {
    if (controller.zoom.changed) {
        camera.focal_length += controller.zoom.in * delta_time;
        
        generateRayDirections();
        rotateRayDirections();
        
        onMouseWheelChangeHandled();
    }

    if (controller.rotation.changed) {
        rotate(&camera.matrix,
                controller.rotation.yaw * delta_time,
                controller.rotation.pitch * delta_time,
                0);
        rotateRayDirections();

        onMousePositionChangeHandled();
    }

    if (keyboard.pressed)
        processKeyboardInputs(delta_time);
}

void init_spheres(u8 radius, u8 horizontal_count, u8 vertical_count) {
    sphere_count = horizontal_count * vertical_count;
    sphere_positions = (Vector3*)allocate_memory(sizeof(Vector3) * sphere_count);
    
    sphere.position = sphere_positions;
    sphere.radius = radius;

    u8 gap = radius * 3;
    u8 sphere_x = 0;
    u8 sphere_z = 0;

    for (u8 z = 0; z < vertical_count; z++) {
        sphere_x = 0;

        for (u8 x = 0; x < horizontal_count; x++) {
            sphere.position->x = sphere_x;
            sphere.position->y = 0;
            sphere.position->z = sphere_z;

            sphere_x += gap;
            sphere.position++;
        }

        sphere_z += gap;
    }
}


void init_renderer() {
    init_math3D();
    init_core3D();
    init_spheres(1, 2, 2);

    frame_buffer.pixels = (Pixel*)allocate_memory(RENDER_SIZE);
    
    source_ray_directions = (Vector3*)allocate_memory(RENDER_SIZE);
    ray_directions = (Vector3*)allocate_memory(RENDER_SIZE);
    
    ray_origin_to_sphere_center = (Vector3*)allocate_memory(sizeof(Vector3));
    ray_origin_to_closest_position = (Vector3*)allocate_memory(sizeof(Vector3));

    closest_position_to_sphere_center = (Vector3*)allocate_memory(sizeof(Vector3));
    closest_hit_surface_normal = (Vector3*)allocate_memory(sizeof(Vector3));
    
    camera.position = ray.origin = (Vector3*)allocate_memory(sizeof(Vector3));
    camera.position->x = 5;
    camera.position->y = 5;
    camera.position->z = -10;
}