#pragma once

#define initShader() \
    initKernel();    \
                         \
    Pixel *pixel = (Pixel *)&d_pixels[i]; \
                     \
    vec3 ray_origin = d_vectors[0];    \
    vec3 ray_direction = d_vectors[1];    \
    vec3 right = d_vectors[2];            \
    vec3 down = d_vectors[3];\
    Ray ray;         \
    ray.origin = &ray_origin; \
    ray.direction = &ray_direction;   \
    iscaleVec3(&right, x); iaddVec3(ray.direction, &right); \
    iscaleVec3(&down,  y); iaddVec3(ray.direction, &down); \
    norm3(ray.direction);                 \
    ray.hit.distance = MAX_DISTANCE; \
                         \
    Scene scene;         \
    scene.materials = d_materials; \
    scene.point_lights = d_point_lights; \
    scene.tetrahedra = d_tetrahedra; \
    scene.spheres = d_spheres; \
    scene.planes = d_planes; \
    scene.cubes = d_cubes; \
    scene.ambient_light = d_ambient_light;\
    scene.cube_indices = d_cube_indices;\
    scene.tetrahedron_indices = d_tetrahedron_indices

__global__ void d_renderUVs() {     initShader(); renderUVs(     &ray, &scene, d_bvh_nodes, d_ssb_bounds, d_masks, x, y, pixel); }
__global__ void d_renderDepth() {   initShader(); renderDepth(   &ray, &scene, d_bvh_nodes, d_ssb_bounds, d_masks, x, y, pixel); }
__global__ void d_renderBeauty() {  initShader(); renderBeauty(  &ray, &scene, d_bvh_nodes, d_ssb_bounds, d_masks, x, y, pixel); }
__global__ void d_renderNormals() { initShader(); renderNormals( &ray, &scene, d_bvh_nodes, d_ssb_bounds, d_masks, x, y, pixel); }

void renderOnGPU(vec3 *Ro, vec3 *start, vec3 *right, vec3 *down) {
    setupKernel()

    vec3 vectors[4];
    vectors[0] = *Ro;
    vectors[1] = *start;
    vectors[2] = *right;
    vectors[3] = *down;

    gpuErrchk(cudaMemcpyToSymbol(d_vectors, vectors, sizeof(vec3) * 4, 0, cudaMemcpyHostToDevice));

    switch (render_mode) {
        case Beauty    : d_renderBeauty<<< blocks, threads>>>(); break;
        case Depth     : d_renderDepth<<<  blocks, threads>>>(); break;
        case Normals   : d_renderNormals<<<blocks, threads>>>(); break;
        case UVs       : d_renderUVs<<<    blocks, threads>>>(); break;
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaMemcpyFromSymbol((u32*)frame_buffer.pixels, d_pixels, sizeof(u32) * frame_buffer.dimentions.width_times_height, 0, cudaMemcpyDeviceToHost));
}