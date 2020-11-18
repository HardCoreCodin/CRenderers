#pragma once

#include "lib/core/types.h"
#include "lib/core/hud.h"
#include "lib/core/perf.h"
#include "lib/core/text.h"
#include "lib/input/mouse.h"
#include "lib/input/keyboard.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/controllers/camera_controller.h"
#include "lib/nodes/scene.h"
#include "lib/nodes/camera.h"
#include "lib/memory/buffers.h"
#include "lib/memory/allocators.h"
#include "lib/render/raytracer.h"

UpdateWindowTitle updateWindowTitle;
PrintDebugString printDebugString;

char* getTitle() {
    return RAY_TRACER_TITLE;
}

#define SPHERE_TURN_SPEED 0.3f
#define TETRAHEDRON_TURN_SPEED 0.3f

Sphere *rotating_sphere;
Tetrahedron *rotating_tetrahedron, ref_tetrahedron;

void updateAndRender() {
    startFrameTimer(&update_timer);

    yawMat3(update_timer.delta_time * SPHERE_TURN_SPEED, &rotating_sphere->rotation_matrix);
    yawMat3(update_timer.delta_time * TETRAHEDRON_TURN_SPEED, &rotating_tetrahedron->rotation_matrix);
    vec3 *ref_vertex = ref_tetrahedron.vertices,
         *tet_vertex = rotating_tetrahedron->vertices,
         *position = &rotating_tetrahedron->position;
    mat3 *rotation = &rotating_tetrahedron->rotation_matrix;
    Triangle *ref_triangle = ref_tetrahedron.triangles,
             *tet_triangle = rotating_tetrahedron->triangles;
    for (u8 i = 0; i < 4; i++, ref_vertex++, tet_vertex++, ref_triangle++, tet_triangle++) {
        mulVec3Mat3(ref_vertex, rotation, tet_vertex);
        iaddVec3(tet_vertex, position);
        mulMat3(&ref_triangle->tangent_to_world, rotation, &tet_triangle->tangent_to_world);
        transposeMat3(&tet_triangle->tangent_to_world, &tet_triangle->world_to_tangent);
    }

    if (mouse_wheel_scrolled) current_camera_controller->onMouseWheelScrolled();
    if (mouse_moved)          current_camera_controller->onMouseMoved();
    current_camera_controller->onUpdate();

    if (current_camera_controller->zoomed) onZoom();
    if (current_camera_controller->turned) onTurn();
    if (current_camera_controller->moved)  onMove();

    onRender();

    endFrameTimer(&update_timer);
    if (hud.is_visible) {
        if (!update_timer.accumulated_frame_count) updateHUDCounters(&update_timer);
        drawText(&frame_buffer, hud.text, HUD_COLOR, frame_buffer.width - HUD_RIGHT - HUD_WIDTH, HUD_TOP);
    }

    if (mouse_double_clicked) {
        mouse_double_clicked = false;
        bool in_fps_mode = current_camera_controller == &fps_camera_controller.controller;
        current_camera_controller = in_fps_mode ?
                                    &orb_camera_controller.controller :
                                    &fps_camera_controller.controller;
        setControllerModeInHUD(!in_fps_mode);
    }
}

void resize(u16 width, u16 height) {
    frame_buffer.width = width;
    frame_buffer.height = height;
    frame_buffer.size = frame_buffer.width * frame_buffer.height;
    frame_buffer.width_over_height = (f32)frame_buffer.width / (f32)frame_buffer.height;
    frame_buffer.height_over_width = (f32)frame_buffer.height / (f32)frame_buffer.width;

    onResize();
    updateHUDDimensions();
    updateAndRender();
}

void initEngine(
    UpdateWindowTitle platformUpdateWindowTitle,
    PrintDebugString platformPrintDebugString,
    GetTicks platformGetTicks,
    u64 platformTicksPerSecond
) {
    updateWindowTitle = platformUpdateWindowTitle;
    printDebugString  = platformPrintDebugString;
    initTimers(platformGetTicks, platformTicksPerSecond);
    initHUD();
    initFrameBuffer();
    initScene();
    initFpsController(scene.camera);
    initOrbController(scene.camera);
    initRayTracer();
    initTetrahedron(&ref_tetrahedron);
    rotating_tetrahedron = scene.tetrahedra;
    rotating_sphere = scene.spheres + 1;

    scene.camera->transform.position.x = 5;
    scene.camera->transform.position.y = 5;
    scene.camera->transform.position.z = -10;
    current_camera_controller = &orb_camera_controller.controller;
    current_camera_controller->turned = true;
}

//void draw_line(int x0, int y0, int x1, int y1, u32 color) {
//    int index = x0 + y0 * window_width;
//    if (x0 == x1 && y0 == y1) { // Draw single pixel:
//        DRAW_PIXEL(index, color);
//        return;
//    }
//
//    int dx = 1;
//    int dy = 1;
//    int run  = x1 - x0;
//    int rise = y1 - y0;
//    if (x0 > x1) {
//        dx = -1;
//        run  = x0 - x1;
//    }
//
//    int index_inc_per_line = window_width;
//    if (y0 > y1) {
//        dy = -1;
//        rise = y0 - y1;
//        index_inc_per_line = -window_width;
//    }
//
//    // Configure for a trivial line (horizontal, vertical or diagonal, default to a shallow line):
//    int inc = dx;
//    int start = x0;
//    int end = x1 + inc;
//    int index_inc = dx;
//    if (rise > run) { // Reconfigure for a steep line:
//        inc = dy;
//        start = y0;
//        end = y1 + inc;
//        index_inc = index_inc_per_line;
//    }
//
//    if (rise == run || !rise || !run) { // Draw a trivial line:
//        if (rise && run) // Reconfigure for a diagonal line:
//            index_inc = index_inc_per_line + dx;
//
//        for (int i = start; i != end; i += inc, index += index_inc)
//            DRAW_PIXEL(index, color);
//
//        return;
//    }
//
//    // Configure for a non-trivial line (default to a shallow line):
//    int rise_twice = rise + rise;
//    int run_twice  = run + run;
//    int threshold = run;
//    int error_dec = run_twice;
//    int error_inc = rise_twice;
//    int secondary_inc = index_inc_per_line;
//    if (rise > run) { // Reconfigure for a steep line:
//        secondary_inc = dx;
//        threshold = rise;
//        error_dec = rise_twice;
//        error_inc = run_twice;
//    }
//
//    int error = 0;
//    for (int i = start; i != end; i += inc) {
//        DRAW_PIXEL(index, color);
//        index += index_inc;
//        error += error_inc;
//        if (error > threshold) {
//            error -= error_dec;
//            index += secondary_inc;
//        }
//    }
//}
//
//
//void drawTriangle(float* X, float* Y, int pitch, u32 color, u32* pixels) {
//    float dx1, x1, y1, xs,
//          dx2, x2, y2, xe,
//          dx3, x3, y3, dy;
//    int offset,
//        x, x1i, y1i, x2i, xsi, ysi = 0,
//        y, y2i, x3i, y3i, xei, yei = 0;
//    for (int i = 1; i <= 2; i++) {
//        if (Y[i] < Y[ysi]) ysi = i;
//        if (Y[i] > Y[yei]) yei = i;
//    }
//    byte* id = ysi ? (ysi == 1 ?
//            (byte[3]){1, 2, 0} :
//            (byte[3]){2, 0, 1}) :
//            (byte[3]){0, 1, 2};
//    x1 = X[id[0]]; y1 = Y[id[0]]; x1i = (int)x1; y1i = (int)y1;
//    x2 = X[id[1]]; y2 = Y[id[1]]; x2i = (int)x2; y2i = (int)y2;
//    x3 = X[id[2]]; y3 = Y[id[2]]; x3i = (int)x3; y3i = (int)y3;
//    dx1 = x1i == x2i || y1i == y2i ? 0 : (x2 - x1) / (y2 - y1);
//    dx2 = x2i == x3i || y2i == y3i ? 0 : (x3 - x2) / (y3 - y2);
//    dx3 = x1i == x3i || y1i == y3i ? 0 : (x3 - x1) / (y3 - y1);
//    dy = 1 - (y1 - (float)y1);
//    xs = dx3 ? x1 + dx3 * dy : x1; ysi = (int)Y[ysi];
//    xe = dx1 ? x1 + dx1 * dy : x1; yei = (int)Y[yei];
//    offset = pitch * y1i;
//    for (y = ysi; y < yei; y++){
//        if (y == y3i) xs = dx2 ? (x3 + dx2 * (1 - (y3 - (float)y3i))) : x3;
//        if (y == y2i) xe = dx2 ? (x2 + dx2 * (1 - (y2 - (float)y2i))) : x2;
//        xsi = (int)xs;
//        xei = (int)xe;
//        for (x = xsi; x < xei; x++) pixels[offset + x] = color;
//        offset += pitch;
//        xs += y < y3i ? dx3 : dx2;
//        xe += y < y2i ? dx1 : dx2;
//    }
//}
//
//float triangles_x[3] = {120.7f, 220.3f, 320.4f};
//float triangles_y[3] = {200.5f, 158.2f, 200.6f};