#pragma once

#include "lib/core/types.h"
#include "lib/core/hud.h"
#include "lib/core/perf.h"
#include "lib/core/text.h"

#include "lib/globals/app.h"
#include "lib/globals/timers.h"
#include "lib/globals/inputs.h"
#include "lib/globals/camera.h"
#include "lib/globals/scene.h"
#include "lib/globals/display.h"

#include "lib/input/mouse.h"
#include "lib/input/keyboard.h"

#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/controllers/camera_controller.h"

#include "lib/nodes/scene.h"
#include "lib/nodes/camera.h"

#include "lib/shapes/helix.h"

#include "lib/memory/allocators.h"

#include "lib/render/raytracer.h"

char* getTitle() {
    return RAY_TRACER_TITLE;
}

#define SPHERE_TURN_SPEED 0.3f
#define TETRAHEDRON_TURN_SPEED 0.3f

Tetrahedron *rotating_tetrahedron;

bool use_old_SSB_before = false;

void updateAndRender() {
    use_GPU = shift_is_pressed;
    use_BVH = space_is_pressed;
//    use_old_SSB = space_is_pressed;
    render_mode = alt_is_pressed ? UVs : (ctrl_is_pressed ? Normal : Beauty);
//    render_mode = Beauty;

    setUseBVH(use_BVH);
//    setUseOld(use_old_SSB);
    setRunOnInHUD(use_GPU);
//    setAltModeInHUD(alt_is_pressed);
    startFrameTimer(&update_timer);

    yawMat3(update_timer.delta_time * SPHERE_TURN_SPEED, &main_scene.spheres[1].rotation);
#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_sphere_rotations, main_scene.sphere_rotations, sizeof(mat3) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
#endif

    if (mouse_wheel_scrolled) {
       if (shift_is_pressed) {
           updateTetrahedronRadius(rotating_tetrahedron, rotating_tetrahedron->radius + mouse_wheel_scroll_amount / 1000);
           mouse_wheel_scroll_amount = 0;
           mouse_wheel_scrolled = false;
           vec3* p = ray_tracer.ssb.view_positions.tetrahedra;
           Bounds2Di *b = ray_tracer.ssb.bounds.tetrahedra;
           computeSSB(b, p->x, p->y, p->z, rotating_tetrahedron->radius, main_camera.focal_length);
       } else
//        if (ctrl_is_pressed) {
//            my_helix.radius += mouse_wheel_scroll_amount / 1000;
//            mouse_wheel_scroll_amount = 0;
//            mouse_wheel_scrolled = false;
//        } else if (alt_is_pressed) {
//            my_helix.thickness_radius += mouse_wheel_scroll_amount / 1000;
//            mouse_wheel_scroll_amount = 0;
//            mouse_wheel_scrolled = false;
//        } else if (shift_is_pressed) {
//            my_helix.revolution_count += (u32)((f32)mouse_wheel_scroll_amount / 50.0f);
//            mouse_wheel_scroll_amount = 0;
//            mouse_wheel_scrolled = false;
//        } else
            current_camera_controller->onMouseWheelScrolled();
    }

//    if (mouse_wheel_scrolled) {
//        if (ctrl_is_pressed) {
//            my_coil.height += mouse_wheel_scroll_amount / 1000;
//            mouse_wheel_scroll_amount = 0;
//            mouse_wheel_scrolled = false;
//        } else if (alt_is_pressed) {
//            my_coil.radius += mouse_wheel_scroll_amount / 1000;
//            mouse_wheel_scroll_amount = 0;
//            mouse_wheel_scrolled = false;
//        } else if (shift_is_pressed) {
//            my_coil.revolution_count += (u32)((f32)mouse_wheel_scroll_amount / 50.0f);
//            mouse_wheel_scroll_amount = 0;
//            mouse_wheel_scrolled = false;
//        } else
//            current_camera_controller->onMouseWheelScrolled();
//    }

    f32 amount = update_timer.delta_time * TETRAHEDRON_TURN_SPEED;
    xform3 local_xform;
    initXform3(&local_xform);
    rotateXform3(&local_xform, amount, amount/2, amount/3);

    rotateXform3(&rotating_tetrahedron->xform, amount, amount, amount);
    vec3 *tet_vertex = rotating_tetrahedron->vertices,
            *position = &rotating_tetrahedron->xform.position;
    Triangle *tet_triangle = rotating_tetrahedron->triangles;
    imulMat3(&rotating_tetrahedron->xform.rotation_matrix, &local_xform.rotation_matrix);
    transposeMat3(&rotating_tetrahedron->xform.rotation_matrix, &rotating_tetrahedron->xform.rotation_matrix_inverted);
    for (u8 i = 0; i < 4; i++, tet_vertex++, tet_triangle++) {
        isubVec3(tet_vertex, position);
        imulVec3Mat3(tet_vertex, &local_xform.rotation_matrix);
        iaddVec3(tet_vertex, position);

        imulMat3(&tet_triangle->tangent_to_world, &local_xform.rotation_matrix);
    }
    updateTetrahedronMatrices(rotating_tetrahedron);

    if (mouse_moved)          current_camera_controller->onMouseMoved();
    current_camera_controller->onUpdate();

    if (use_old_SSB_before != use_old_SSB) {
        current_camera_controller->moved = true;
        use_old_SSB_before = use_old_SSB;
    }

    if (current_camera_controller->zoomed) onZoom();
    if (current_camera_controller->turned) onTurn();
    if (current_camera_controller->moved)  onMove(&main_scene);

    onRender(&main_scene, &main_camera);

    endFrameTimer(&update_timer, true);
    if (hud.is_visible) {
        if (!update_timer.accumulated_frame_count)
            updateHUDCounters(&update_timer, ray_tracer.stats.visible_nodes[GEO_TYPE__SPHERE-1], ray_tracer.stats.active_pixels);
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
    updateFrameBufferDimensions(width, height);
    onResize(&main_scene);
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
    initAppGlobals();
    initMouse();
    initTimers(platformGetTicks, platformTicksPerSecond);
    initHUD();
    initFrameBuffer();
    initScene(&main_scene);
    initCamera(&main_camera);
    initFpsController(&main_camera);
    initOrbController(&main_camera);
    initRayTracer(&main_scene);
    rotating_tetrahedron = main_scene.tetrahedra;

    main_camera.transform.position.x = 5;
    main_camera.transform.position.y = 5;
    main_camera.transform.position.z = -12;
    current_camera_controller = &orb_camera_controller.controller;
    current_camera_controller->turned = true;
}

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