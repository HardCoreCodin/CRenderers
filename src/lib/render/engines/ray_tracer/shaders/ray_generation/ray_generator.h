#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"

void generateRayDirections(Vector3* ray_directions, f32 focal_length, u16 width, u16 height) {
    f32 ray_direction_length = 0;
    f32 squared_focal_length = focal_length * focal_length;
    f32 one_over_width = 1.0f / (f32)width;
    f32 x_dir = 0, x_dir_squared = 0;
    f32 y_dir = 0, y_dir_squared = 0;

    for (u16 pixel_y = height; pixel_y > 0; pixel_y--) {
        y_dir = (((f32)pixel_y - 0.5f) * one_over_width) - 0.5f;
        y_dir_squared = y_dir * y_dir;

        for (u16 pixel_x = 0; pixel_x < width; pixel_x++) {
            x_dir = (((f32)pixel_x + 0.5f) * one_over_width) - 0.5f;
            x_dir_squared = x_dir * x_dir;

            ray_direction_length = sqrtf(x_dir_squared + y_dir_squared + squared_focal_length);

            ray_directions->x = x_dir / ray_direction_length;
            ray_directions->y = y_dir / ray_direction_length;
            ray_directions->z = focal_length / ray_direction_length;
            ray_directions++;
        }
    }
}

//
//void generateRayDirections() {
//    ray_direction = source_ray_directions;
//
//    f32 ray_direction_length = 0;
//    f32 squared_focal_length = camera.focal_length * camera.focal_length;
//    f32 one_over_width = 1.0f / frame_buffer.width;
//    f32 x, x2;
//
//    for (u16 i = 0; i < frame_buffer.width; i++) {
//        x = ((i + 0.5f) * one_over_width) - 0.5f;
//        x2 = x * x;
//
//        ray_direction_length = sqrtf(x2 + squared_focal_length);
//
//        ray_direction->x = x / ray_direction_length;
//        ray_direction->y = camera.focal_length / ray_direction_length;
//        ray_direction++;
//    }
//
//    ray_directions = ray_direction;
//    rotateRayDirections();
//}

//
//void generateRayDirectionsRat() {
//    Vector3* ray_direction = render_engine.source_ray_directions;
//
//    f32 projection_plane_half_width = 1 / render_engine.core->camera.focal_length;
//
//    f32 w = frame_buffer.width;
//    f32 h = frame_buffer.height;
//    f32 h_over_w = h / w;
//
//    f32 x_step = -projection_plane_half_width / w;
//    f32 y_step = x_step * h_over_w;
//
//    Matrix3x3 yaw, pitch;
//    setMatrix3x3ToIdentity(&yaw);
//    setMatrix3x3ToIdentity(&pitch);
//
//    setYaw3D(-x_step, &yaw);
//    setPitch3D(-y_step, &pitch);
//
//    Vector3 starting_ray_direction = {0, 0, 1};
//    for (int i = 0; i < frame_buffer.height / 2; i++) imul3D(&starting_ray_direction, &pitch);
//    for (int i = 0; i < frame_buffer.width / 2; i++) imul3D(&starting_ray_direction, &yaw);
//
//
////
////    setYaw3D(projection_plane_half_width / 2 + x_step / 2, &yaw);
////
////    setPitch3D(h_over_w * projection_plane_half_width / 2 + y_step / 2, &pitch);
////
////    rotate3D(projection_plane_half_width / 2 + x_step / 2,
////             h_over_w * projection_plane_half_width / 2 + y_step / 2, 0,
////             render_engine.transform);
////    Vector3 starting_ray_direction = render_engine.transform->rotation->k;
////
////    imul3D(&starting_ray_direction, &pitch);
////    imul3D(&starting_ray_direction, &yaw);
//
//    setYaw3D(x_step, &yaw);
//    setPitch3D(y_step, &pitch);
//
//    Vector3 current_direction;
//    //
////    setMatrix3x3ToIdentity(render_engine.transform->yaw);
////    setMatrix3x3ToIdentity(render_engine.transform->pitch);
////    rotate3D(x_step, y_step, 0, render_engine.transform);
//
//    for (u16 pixel_y = 0; pixel_y < frame_buffer.height; pixel_y++) {
//        current_direction.x = starting_ray_direction.x;
//        current_direction.y = starting_ray_direction.y;
//        current_direction.z = starting_ray_direction.z;
//
//        for (u16 pixel_x = 0; pixel_x < frame_buffer.width; pixel_x++) {
////            imul3D(ray_direction, render_engine.transform->yaw);
//            ray_direction->x = current_direction.x;
//            ray_direction->y = current_direction.y;
//            ray_direction->z = current_direction.z;
//            ray_direction++;
//
//            imul3D(&current_direction, &yaw);
//        }
//
////        imul3D(&starting_ray_direction, render_engine.transform->pitch);
//        imul3D(&starting_ray_direction, &pitch);
//    }
//
////    f32 x_step = projection_plane_half_width / w;
////    f32 y_step = -x_step * h_over_w;
////
////    f32 x_start = -projection_plane_half_width / 2 + x_step / 2;
////    f32 y_start = (projection_plane_half_width / 2) * h_over_w  + y_step / 2;
////
////    f32 t_x, t_y = y_start;
////
////    for (u16 pixel_y = 0; pixel_y < frame_buffer.height; pixel_y++) {
////        t_x = x_start;
////
////        for (u16 pixel_x = 0; pixel_x < frame_buffer.width; pixel_x++) {
////            f32 z = -2 * (t_x + t_y) / (t_x * t_x + t_y * t_y + 1);
////            f32 x = t_x * z + 1;
////            f32 y = t_y * z + 1;
////
////            ray_direction->x = x * 0.5f;
////            ray_direction->y = y * 0.5f;
////            ray_direction->z = z * 0.5f;
//////            setPointOnUnitSphere(t_x, t_y, ray_direction);
////            ray_direction++;
////
////            t_x += x_step;
////        }
////
////        t_y += y_step;
////    }
//}
