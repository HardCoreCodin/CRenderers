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

#include "lib/nodes/node.h"
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

void updateAndRender() {
    setRunOnInHUD();
    setRenderModeInHUD();

    startFrameTimer(&update_timer);

    yawMat3(update_timer.delta_time * SPHERE_TURN_SPEED, &main_scene.spheres[1].rotation);

    if (mouse_wheel_scrolled) {
       if (shift_is_pressed) {
           Node *node = &main_scene.tetrahedra->node;
           f32 radius = node->radius + mouse_wheel_scroll_amount / 1000;
           setNodeRadius(node, radius);
           vec3* p = ray_tracer.ssb.view_positions.tetrahedra;
           Bounds2Di *b = ray_tracer.ssb.bounds.tetrahedra;
           computeSSB(b, p->x, p->y, p->z, radius, main_camera.focal_length);

           node = &main_scene.cubes->node;
           radius = node->radius + mouse_wheel_scroll_amount / 1000;
           setNodeRadius(node, radius);
           p = ray_tracer.ssb.view_positions.cubes;
           b = ray_tracer.ssb.bounds.cubes;
           computeSSB(b, p->x, p->y, p->z, main_scene.cubes->node.radius, main_camera.focal_length);

           updateBVH(&ray_tracer.bvh, &main_scene);
           mouse_wheel_scroll_amount = 0;
           mouse_wheel_scrolled = false;
#ifdef __CUDACC__
           copySSBBoundsFromCPUtoGPU(&ray_tracer.ssb.bounds);
           copyBVHNodesFromCPUtoGPU(ray_tracer.bvh.nodes);
#endif
       } else
            current_camera_controller->onMouseWheelScrolled();
    }

    f32 amount = update_timer.delta_time * TETRAHEDRON_TURN_SPEED;
    xform3 local_xform;
    initXform3(&local_xform);
    rotateXform3(&local_xform, amount, amount/2, amount/3);
    rotateNode(&main_scene.cubes->node, &local_xform.rotation_matrix);
    rotateNode(&main_scene.tetrahedra->node, &local_xform.rotation_matrix);

#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_cubes, main_scene.cubes, sizeof(Cube) * CUBE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_spheres, main_scene.spheres, sizeof(Sphere) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(d_tetrahedra, main_scene.tetrahedra, sizeof(Tetrahedron) * TETRAHEDRON_COUNT, 0, cudaMemcpyHostToDevice));
#endif
    if (color_control.is_visible) {
        if (left_mouse_button.is_pressed && !color_control.is_controlled) {
            if (inBounds(&color_control.R, mouse_pos)) {color_control.is_controlled = true; color_control.is_red_controlled = true; }
            if (inBounds(&color_control.G, mouse_pos)) {color_control.is_controlled = true; color_control.is_green_controlled = true; }
            if (inBounds(&color_control.B, mouse_pos)) {color_control.is_controlled = true; color_control.is_blue_controlled = true; }
        }
        if (left_mouse_button.is_released) {
            color_control.is_controlled = false;
            color_control.is_red_controlled = false;
            color_control.is_blue_controlled = false;
            color_control.is_green_controlled = false;
        }
    }

    if (mouse_moved) {
        if (color_control.is_visible && color_control.is_controlled) {
            if (color_control.is_red_controlled) updateRedColorControl(mouse_movement);
            if (color_control.is_green_controlled) updateGreenColorControl(mouse_movement);
            if (color_control.is_blue_controlled) updateBlueColorControl(mouse_movement);
            mouse_movement.x = mouse_movement.y = 0;
#ifdef __CUDACC__
            gpuErrchk(cudaMemcpyToSymbol(d_point_lights, main_scene.point_lights, sizeof(PointLight) * POINT_LIGHT_COUNT, 0, cudaMemcpyHostToDevice));
#endif
        } else
            current_camera_controller->onMouseMoved();
    }
    current_camera_controller->onUpdate();

    if (current_camera_controller->zoomed) onZoom();
    if (current_camera_controller->turned) onTurn();
    if (current_camera_controller->moved)  onMove(&main_scene);

    onRender(&main_scene, &main_camera);

    endFrameTimer(&update_timer, true);
    if (hud.is_visible) {
        if (!update_timer.accumulated_frame_count) setCountersInHUD(&update_timer);
        drawText(&frame_buffer, hud.text, HUD_COLOR, frame_buffer.dimentions.width - HUD_RIGHT - HUD_WIDTH, HUD_TOP);
    }
    if (color_control.is_visible) drawColorControl();

    if (mouse_double_clicked) {
        mouse_double_clicked = false;
        bool in_fps_mode = current_camera_controller == &fps_camera_controller.controller;
        current_camera_controller = in_fps_mode ?
                                    &orb_camera_controller.controller :
                                    &fps_camera_controller.controller;
    }
}

void resize(u16 width, u16 height) {
    updateFrameBufferDimensions(width, height);
    onResize(&main_scene);
    setDimesionsInHUD();
    updateAndRender();
}

void initEngine(
    UpdateWindowTitle platformUpdateWindowTitle,
    PrintDebugString platformPrintDebugString,
    GetTicks platformGetTicks,
    u64 platformTicksPerSecond,
    KeyMap key_map
) {
    keys = key_map;
    updateWindowTitle = platformUpdateWindowTitle;
    printDebugString  = platformPrintDebugString;
    initAppGlobals();
    initMouse();
    initTimers(platformGetTicks, platformTicksPerSecond);
    initFrameBuffer();
    initScene(&main_scene);
    initCamera(&main_camera);
    initFpsController(&main_camera);
    initOrbController(&main_camera);
    initRayTracer(&main_scene);
    initHUD();
    initColorControl();
    bindColorControl(&main_scene.point_lights->color);
    color_control.is_visible = true;
//    initGammaLUT();

    main_camera.transform.position.x = 5;
    main_camera.transform.position.y = 5;
    main_camera.transform.position.z = -12;
    current_camera_controller = &orb_camera_controller.controller;
    current_camera_controller->turned = true;
}