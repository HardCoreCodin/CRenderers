#pragma once

#include "lib/core/types.h"
#include "lib/core/hud.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"

Viewport viewport;

void refreshViewport() {
    viewport.renderer->render(viewport.controller);
}

void resizeViewport() {
    viewport.renderer->resize(viewport.controller);
    updateHUDDimensions();
}

void bindControllerToRenderer() {
    viewport.controller->zoom = viewport.renderer->zoom;
    viewport.controller->move = viewport.renderer->move;
    viewport.controller->rotate = viewport.renderer->rotate;
}

void setViewportRenderer(Renderer* renderer) {
    viewport.renderer = renderer;
    bindControllerToRenderer();
}

void setViewportController(Controller* controller) {
    controller->on.reset();
    viewport.controller = controller;
    bindControllerToRenderer();
}

void toggleControllerMode() {
    viewport.in_fps_mode = !viewport.in_fps_mode;
    setViewportController(viewport.in_fps_mode ? &fps.controller : &orb.controller);
    setControllerModeInHUD(viewport.in_fps_mode);
}

void initViewport(Renderer* renderer, Controller* controller) {
    viewport.camera = controller->camera;
    viewport.renderer = renderer;
    viewport.controller = controller;

    viewport.in_fps_mode = false;

    viewport.resize = resizeViewport;
    viewport.refresh = refreshViewport;
    viewport.setRenderer = setViewportRenderer;
    viewport.setController = setViewportController;
    viewport.toggleControllerMode = toggleControllerMode;

    bindControllerToRenderer();
}