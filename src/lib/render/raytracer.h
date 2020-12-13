#pragma once

#include "lib/core/types.h"
#include "lib/globals/raytracing.h"
#include "lib/shapes/line.h"
#include "lib/shapes/bbox.h"
#include "lib/shapes/helix.h"
#include "lib/input/keyboard.h"
#include "lib/controllers/fps.h"
#include "lib/controllers/orb.h"
#include "lib/controllers/camera_controller.h"
#include "lib/nodes/camera.h"
#include "lib/memory/allocators.h"

#include "lib/render/shaders/closest_hit/debug.h"
#include "lib/render/shaders/closest_hit/classic.h"
#include "lib/render/shaders/closest_hit/surface.h"
#include "lib/render/shaders/intersection/cube.h"
#include "lib/render/shaders/intersection/plane.h"
#include "lib/render/shaders/intersection/sphere.h"
#include "lib/render/shaders/intersection/tetrahedra.h"
#include "lib/render/shaders/ray_generation/primary_rays.h"

#include "BVH.h"
#include "SSB.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool tracePrimaryRay(Ray *ray, Scene *scene, BVH *bvh, SSB *ssb, Masks *masks, u16 x, u16 y, bool using_BVH) {
    ray->hit.uv.x = ray->hit.uv.y = 1;
    ray->hit.distance = MAX_DISTANCE;

//    if (using_BVH)
//        setRayMasksFromBVH(ray, bvh);
//    else
        setRayMasksFromSSB(ray, masks, ssb, x, y);

    ray->masks.shadowing[GEO_TYPE__SPHERE-1] = ray_tracer.masks.shadowing[GEO_TYPE__SPHERE-1];
    ray->masks.transparency[GEO_TYPE__SPHERE-1] = ray_tracer.masks.transparency[GEO_TYPE__SPHERE-1];

    bool found_sphere = false;
    bool found_plane = hitPlanes(scene->planes, ray);
    if (ray->masks.visibility[GEO_TYPE__SPHERE-1])
        found_sphere = hitSpheres(scene->spheres, ray, false);

    ray->masks.shadowing[GEO_TYPE__TETRAHEDRON-1] = ray_tracer.masks.shadowing[GEO_TYPE__TETRAHEDRON-1];
    ray->masks.transparency[GEO_TYPE__TETRAHEDRON-1] = ray_tracer.masks.transparency[GEO_TYPE__TETRAHEDRON-1];
    bool found_tetrahedron = false;
    if (ray->masks.visibility[GEO_TYPE__TETRAHEDRON-1])
        found_tetrahedron = hitTetrahedraSimple(scene->tetrahedra, ray);

    return found_plane | found_sphere | found_tetrahedron;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else

inline
#endif
bool traceSecondaryRay(Ray *ray, Scene *scene, BVH *bvh, Masks *masks) {
    ray->hit.uv.x = ray->hit.uv.y = 1;
    ray->hit.distance = MAX_DISTANCE;
    for (u8 i = 0; i < GEO_TYPE__NONE; i++) {
        ray->masks.transparency[i] = masks->transparency[i];
        ray->masks.visibility[i] = FULL_MASK;
    }

    bool found_plane  = hitPlanes(  scene->planes, ray);
    bool found_sphere = hitSpheres(scene->spheres, ray, false);

    return found_plane | found_sphere;
}


#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
bool traceSecondaryRaySimple(Ray *ray, Scene *scene, BVH *bvh, Masks *masks) {
    ray->hit.uv.x = ray->hit.uv.y = 1;
    ray->hit.distance = MAX_DISTANCE;
    for (u8 i = 0; i < GEO_TYPE__NONE; i++) {
        ray->masks.transparency[i] = masks->transparency[i];
        ray->masks.visibility[i] = FULL_MASK;
    }
    bool found_plane  = hitPlanes(        scene->planes, ray);
    bool found_sphere = hitSpheresSimple(scene->spheres, ray);

    return found_plane | found_sphere;
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void renderPixel(Ray *ray, Scene *scene, BVH *bvh, SSB *ssb, Masks *masks, u16 x, u16 y, bool using_BVH, Pixel* pixel) {
    tracePrimaryRay(ray, scene, bvh, ssb, masks, x, y, using_BVH);

    vec3 color;
    fillVec3(&color, 0);
//    shadeSurface(scene, ray->hit.material_id, ray->direction,  &ray->hit.position, &ray->hit.normal, &color);
    shadeLambert(scene, ray->direction, &ray->hit.position, &ray->hit.normal, &color);
//    shadePhong(scene, ray->direction, &ray->hit.position, &ray->hit.normal &color);
//    shadeBlinn(scene, ray->direction, &ray->hit.position, &ray->hit.normal, &color);
//    shadeDepth(ray->hit.distance, &color);
    setPixelColor(pixel, color);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void renderNorm(Ray *ray, Scene *scene, BVH *bvh, SSB *ssb, Masks *masks, u16 x, u16 y, bool using_BVH, Pixel* pixel) {
    tracePrimaryRay(ray, scene, bvh, ssb, masks, x, y, using_BVH);

    vec3 color;
    shadeDirection(&ray->hit.normal, &color);
    setPixelColor(pixel, color);
}

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void renderUV(Ray *ray, Scene *scene, BVH *bvh, SSB *ssb, Masks *masks, u16 x, u16 y, bool using_BVH, Pixel* pixel) {
    tracePrimaryRay(ray, scene, bvh, ssb, masks, x, y, using_BVH);

    vec3 color;
    shadeUV(ray->hit.uv, &color);
    setPixelColor(pixel, color);
}
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//void renderVisNew(Scene *scene, BVH *bvh, Ray *ray, Pixel* pixel) {
//    traverseBVH(bvh, ray);
//    pixel->color.A = 0;
//    pixel->color.R = pixel->color.G = pixel->color.B = ray->visibility_masks[0] ? MAX_COLOR_VALUE : 0;
//}
//
//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//void renderVis(Scene *scene, Ray *ray, u16 x, u16 y, Pixel* pixel) {
//    ray->visibility_masks[0] = getSphereVisibilityMask(scene->sphere_view_bounds, scene->masks->visibility, x, y);
//    pixel->color.A = 0;
//    pixel->color.R = pixel->color.G = pixel->color.B = ray->visibility_masks[0] ? MAX_COLOR_VALUE : 0;
//}
//
#ifdef __CUDACC__

#define initShader(C, W) \
    initKernel(C, W); \
    Pixel *pixel = (Pixel *)&d_pixels[i]; \
    Ray ray; \
    ray.direction = &d_ray_directions[i]; \
    ray.origin = &Ro; \
    ray.hit.distance = MAX_DISTANCE; \
    Ray *R = &ray;   \
    Scene scene; \
    scene.masks = d_masks; \
    scene.materials = d_materials; \
    scene.point_lights = d_point_lights; \
    scene.tetrahedra = d_tetrahedra; \
    scene.spheres = d_spheres; \
    scene.planes = d_planes; \
    scene.cubes = d_cubes; \
    scene.ambient_light = d_ambient_light; \
    scene.sphere_view_positions = d_sphere_view_positions; \
    scene.sphere_view_bounds = d_sphere_view_bounds; \
    scene.sphere_rotations = d_sphere_rotations; \
    Scene *S = &scene

__global__ void d_renderVis(  u16 W, u32 C, vec3 Ro) { initShader(C, W); d_masks_out[i] = renderVis(  S, R, x, y, pixel); }
__global__ void d_renderNorm( u16 W, u32 C, vec3 Ro) { initShader(C, W); d_masks_out[i] = renderNorm( S, R, x, y, pixel); }
__global__ void d_renderUVs(  u16 W, u32 C, vec3 Ro) { initShader(C, W); d_masks_out[i] = renderUV(   S, R, x, y, pixel); }
__global__ void d_renderPixel(u16 W, u32 C, vec3 Ro) { initShader(C, W); d_masks_out[i] = renderPixel(S, R, x, y, pixel); }

void renderOnGPU(Scene *scene, Camera *camera) {
    u32 count = frame_buffer.size;
    setupKernel(count)

    vec3 Ro = camera->transform.position;

    switch (render_mode) {
        case Visibility:   d_renderVis<<<blocks, threads>>>(frame_buffer.width, count, Ro); break;
        case Beauty    : d_renderPixel<<<blocks, threads>>>(frame_buffer.width, count, Ro); break;
        case Normal    :  d_renderNorm<<<blocks, threads>>>(frame_buffer.width, count, Ro); break;
        case UVs       :   d_renderUVs<<<blocks, threads>>>(frame_buffer.width, count, Ro); break;
    }

    copyPixelsFromGPUtoCPU((u32*)frame_buffer.pixels, count);
    copyMasksFromGPUtoCPU(ray_tracer.sphere_visibility_masks, count);

    frame_buffer.active_pixel_count = 0;
    for (u32 i = 0; i < frame_buffer.size; i++)
        if (ray_tracer.sphere_visibility_masks[i])
            frame_buffer.active_pixel_count++;
}
#endif

#define runShader(shader, using_BVH) { \
    for (u16 y = 0; y < frame_buffer.height; y++) { \
        for (u16 x = 0; x < frame_buffer.width; x++, pixel++, ray.direction++, ray.direction_rcp++) { \
            shader(&ray, scene, bvh, ssb, masks, x, y, using_BVH, pixel); \
            if (ray.masks.visibility[0]) active_pixels++; \
        }  \
    }  \
}

void renderOnCPU(Scene *scene, Camera *camera) {
    Pixel* pixel = frame_buffer.pixels;
    Masks *masks = &ray_tracer.masks;
    SSB *ssb = &ray_tracer.ssb;
    BVH *bvh = &ray_tracer.bvh;
//    BVH_SoA *bvh = &ray_tracer.bvh_soa;
    u32 active_pixels = 0;

    Ray ray;
    ray.origin = &camera->transform.position;
    ray.direction = ray_tracer.ray_directions;
    ray.direction_rcp = ray_tracer.ray_directions_rcp;

    switch (render_mode) {
        case Beauty    : runShader(renderPixel, use_BVH) break;
        case Normal    : runShader(renderNorm, use_BVH)  break;
        case UVs       : runShader(renderUV, use_BVH)    break;
    }



    ray_tracer.stats.active_pixels = active_pixels;
}

void onZoom() {
    generateRayDirections();
    current_camera_controller->moved = true;
    current_camera_controller->zoomed = false;
}

void onTurn() {
    generateRayDirections();
    transposeMat3(&current_camera_controller->camera->transform.rotation_matrix,
                  &current_camera_controller->camera->transform.rotation_matrix_inverted);
    current_camera_controller->turned = false;
    current_camera_controller->moved = true;
}

void onMove(Scene* scene) {
    vec3 *position = &current_camera_controller->camera->transform.position;
    mat3 *rotation = &current_camera_controller->camera->transform.rotation_matrix_inverted;
    for (u8 i = 0; i < SPHERE_COUNT; i++) {
        subVec3(&scene->spheres[i].position, position, &ray_tracer.ssb.view_positions[i]);
        imulVec3Mat3(&ray_tracer.ssb.view_positions[i], rotation);
    }
    for (u8 i = 0; i < TETRAHEDRON_COUNT; i++) {
        subVec3(&scene->tetrahedra[i].xform.position, position, &ray_tracer.ssb.view_positions[SPHERE_COUNT+i]);
        imulVec3Mat3(&ray_tracer.ssb.view_positions[SPHERE_COUNT+i], rotation);
    }
#ifdef __CUDACC__
    gpuErrchk(cudaMemcpyToSymbol(d_sphere_view_positions, scene->sphere_view_positions, sizeof(vec3) * SPHERE_COUNT, 0, cudaMemcpyHostToDevice));
#endif

//    if (use_old_SSB)
//        updateSceneMasksOld(scene);
//    else
        updateSceneMasks(scene, &ray_tracer.ssb, &ray_tracer.masks, current_camera_controller->camera->focal_length);

    current_camera_controller->moved = false;
}

void onResize(Scene *scene) {
    generateRayDirections();
    onMove(scene);
}

void onRender(Scene *scene, Camera *camera) {

#ifdef __CUDACC__
    if (use_GPU) {
        if (last_rendered_on_CPU) generateRays();

        renderOnGPU(scene, camera);

        last_rendered_on_CPU = false;
    } else {
        if (!last_rendered_on_CPU) generateRays();

        renderOnCPU(scene, camera);

        last_rendered_on_CPU = true;
    }
#else
    renderOnCPU(scene, camera);
#endif
    Pixel pixel;
    pixel.color.R = MAX_COLOR_VALUE;
    pixel.color.G = use_old_SSB ? 0 : MAX_COLOR_VALUE;
    pixel.color.B = 0;
    pixel.color.A = 0;

//    if (show_BVH) drawBVH(camera);
//    if (show_BVH) drawBVH_SoA(camera);
    if (show_BVH) drawBVH(&ray_tracer.bvh, camera);
    if (show_SSB) drawSSB(&ray_tracer.ssb, &pixel);

//    drawHelix(camera, &my_helix, &my_helix_pixel);
    drawCoil(camera, &my_coil, &my_helix_pixel);
}


void initRayTracer(Scene *scene) {
    initBVH(&ray_tracer.bvh, 3); updateBVH(&ray_tracer.bvh, scene);
//    initBVH(&ray_tracer.bvh, scene);
//    initBVH_SoA(&ray_tracer.bvh_soa, 2, 4);
//    updateBVH_SoA(&ray_tracer.bvh_soa, scene);
//    ray_tracer.bvh_soa.depth = 1;

    ray_tracer.rays_per_pixel = 1;
    ray_tracer.ray_count = ray_tracer.rays_per_pixel * MAX_WIDTH * MAX_HEIGHT;
    ray_tracer.ray_directions = AllocN(vec3, ray_tracer.ray_count);
    ray_tracer.ray_directions_rcp = AllocN(vec3, ray_tracer.ray_count);

    ray_tracer.masks.shadowing[GEO_TYPE__SPHERE-1] = 0;
    ray_tracer.masks.visibility[GEO_TYPE__SPHERE-1] = 0;
    ray_tracer.masks.transparency[GEO_TYPE__SPHERE-1] = 0;

    u8 sphere_id = 1;
    for (u8 i = 0; i < SPHERE_COUNT; i++, sphere_id <<= (u8)1) {
        if (i != 3) ray_tracer.masks.shadowing[GEO_TYPE__SPHERE-1] |= sphere_id;
        if (scene->materials[scene->spheres[i].material_id].uses & (u8) TRANSPARENCY)
            ray_tracer.masks.transparency[GEO_TYPE__SPHERE-1] |= sphere_id;
        ray_tracer.masks.visibility[GEO_TYPE__SPHERE-1] |= sphere_id;
    }

    ray_tracer.masks.shadowing[GEO_TYPE__TETRAHEDRON-1] = 1;
    ray_tracer.masks.visibility[GEO_TYPE__TETRAHEDRON-1] = 1;
    ray_tracer.masks.transparency[GEO_TYPE__TETRAHEDRON-1] = 0;

//
//    my_helix.radius = 4;
//    my_helix.thickness_radius = 1;
//    my_helix.revolution_count = 60;
//    my_helix.position.x = my_helix.position.z = 0;
//    my_helix.position.y = 1;
//    my_helix_pixel.color.R = 0;
//    my_helix_pixel.color.G = MAX_COLOR_VALUE;
//    my_helix_pixel.color.B = 0;
//    my_helix_pixel.color.A = 0;
//
//
//    my_coil.radius = 1;
//    my_coil.height = 4;
//    my_coil.revolution_count = 60;
//    my_coil.position.x = my_helix.position.z = 0;
//    my_coil.position.y = 1;
}


//#ifdef __CUDACC__
//__device__
//__host__
//__forceinline__
//#else
//inline
//#endif
//u8 renderPixel(Pixel* pixel, u16 x, u16 y, Ray *ray, Scene *scene) {
//    u8 sphere_visibility_mask = getSphereVisibilityMask(scene->spheres, scene->masks->visibility, x, y);
//
//    vec3 *Rd = ray->direction,
//            *N = &ray->hit.normal,
//            *P = &ray->hit.position;
//
//    ray->hit.distance = MAX_DISTANCE;
//    tracePrimaryRay(ray, scene, sphere_visibility_mask);
//
//    vec3 color, RLd;
//    fillVec3(&color, 0);
//    Material* hit_material = &scene->materials[ray->hit.material_id];
//
//    f32 NdotRd = dotVec3(N, Rd);
//    bool from_behind = NdotRd > 0;
//    NdotRd = -saturate(from_behind ? NdotRd : -NdotRd);
//    if (from_behind) iscaleVec3(N, -1);
//    reflect(Rd, N, NdotRd, &RLd);
//
//    shadeSurface(scene, hit_material, Rd,  &RLd, P, N, &color);

//    vec3 *N, *Rd, *P, L, H, color, scaled_light_color, ambient_color = scene->aux->ambient_color;
//    Material *material;
//    u8 uses, exp;
//    bool from_behind,
//         using_phong,
//         using_blinn,
//         has_diffuse,
//         has_specular,
//         has_reflection,
//         has_refraction,
//         has_rfl_or_rfr;
//    f32 di, si, li, NdotRd,
//        reflection_amount,
//        light_distance_squared,
//        light_distance,
//        diffuse_light,
//        specular_light;
//
//    // Shade primary rays
//    hitPlanes(scene->planes, &hit);
//    if (sphere_visibility_mask)
//        hitSpheresSimple(scene->spheres, true, sphere_visibility_mask, &hit);
//
//    RayHit hit;
//    initRayHit(hit, ray_origin);
//    hit.ray_direction = ray_direction;
//
//    bool reflected = false;
//    bool refracted = false;
//    RayHit refracted_hit,
//           reflected_hit;
//    vec3 *RLd = &reflected_hit.ray_direction;
//    vec3 *RRd = &refracted_hit.ray_direction;
//
//    for (u8 hit_depth = 0; hit_depth < MAX_HIT_DEPTH; hit_depth++) {
//        hitPlanes(scene->planes, &hit);
//        if (sphere_visibility_mask)
//            hitSpheresSimple(scene->spheres, !hit_depth, sphere_visibility_mask, &hit);
//
//        material = &scene->materials[hit.material_id];
//        uses = material->uses;
//
//        using_phong = uses & (u8)PHONG;
//        using_blinn = uses & (u8)BLINN;
//        has_diffuse = uses & (u8)LAMBERT;
//        has_specular = using_phong || using_blinn;
//        has_reflection = uses & (u8)REFLECTION;
//        has_refraction = uses & (u8)REFRACTION;
//
//        di = material->diffuse_intensity;
//        si = material->specular_intensity;
//        exp = material->specular_exponent * (using_blinn ? (u8)4 : (u8)1);
//
//        N = &hit.normal,
//        Rd = &hit.ray_direction,
//        P = &hit.position;
//
//        if (has_reflection || has_refraction || using_phong) {
//            NdotRd = dotVec3(N, Rd);
//            from_behind = NdotRd > 0;
//            NdotRd = -saturate(from_behind ? NdotRd : -NdotRd);
//            if (from_behind) iscaleVec3(N, -1);
//            reflect(Rd, N, NdotRd, RLd);
//        }
//
//        if (has_diffuse || has_specular){
//            color = ambient_color;
//        }
//
//        if (has_reflection || has_refraction) {
//            color.x = color.y = color.z = 0;
//
//            if (has_reflection) {
//                reflected = true;
//                initRayHit(reflected_hit, hit.position);
//            }
//
//            if (has_refraction) {
//                refracted = true;
//                initRayHit(refracted_hit, hit.position);
//                refract(Rd, N, NdotRd, from_behind ? scene->aux->n2_over_n1 : scene->aux->n1_over_n2, RRd);
//            }
//
//            if (has_reflection && has_refraction) {
//                reflection_amount = schlickFresnel(from_behind ? IOR_GLASS : IOR_AIR, from_behind ? IOR_AIR : IOR_GLASS, NdotRd);
////                iscaleVec3(&reflected_color, reflection_amount);
////                iscaleVec3(&refracted_color, 1 - reflection_amount);
//            }
//        } else break;
//    }
//
//
//    if (has_reflection) iaddVec3(out_color, &reflected_color);
//    if (has_refraction) iaddVec3(out_color, &refracted_color);
//
//    vec3 color; color.x = color.y = color.z = 0;
//    if (scene->materials[hit->material_id].uses & (u8)PHONG)
//        shadePhong(scene, hit, &color);
//    else
//        shadeLambert(scene, hit, &color);
////    else shade(scene, &hit, &color);
////                hitCubes(&hit);
//
////                perfStart(&aux_timer);
////                if (shade_normals)
////                    hitImplicitTetrahedra(scene->tetrahedra, &hit);
////                else
////                    hitTetrahedra(scene->tetrahedra, &hit);
////                perfEnd(&aux_timer, aux_timer.accumulated_ticks >= ticks_per_second, i == frame_buffer.size);
//
////        shadeLambert(scene, &hit, &color);
//
//    setPixelColor(pixel, color);
//    return sphere_visibility_mask;
//}