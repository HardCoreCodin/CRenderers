#pragma once

#include <math.h>

#include "lib/core/types.h"
#include "lib/math/math3D.h"
#include "lib/nodes/scene.h"

typedef struct {
    f32 inner_hit_distance, outer_hit_distance;
    bool hit;
} SphereHit;
SphereHit sphere_hits[SPHERE_COUNT];

#define NORMAL_THRESHOLD 0.70710678118654746f

inline void getSphereHitPosition(vec3 *origin, vec3 *direction, f32 distance, vec3 *position) {
    scaleVec3(direction, distance, position);
    iaddVec3(position, origin);
}

inline void getSphereHitNormal(vec3 *position, vec3 *center, vec3 *normal) {
    subVec3(position, center, normal);
    norm3(normal);
}

inline void getSphereHitUV(vec3 *normal, vec2 *uv) {
    f32 factor;
    if (normal->x > +NORMAL_THRESHOLD) { // right:
        factor = 0.5f / normal->x;
        uv->x = 0.5f + factor*normal->z;
        uv->y = 0.5f + factor*normal->y;
    } else if (normal->x < -NORMAL_THRESHOLD) { // left:
        factor = -0.5f / normal->x;
        uv->x = 1 - (0.5f + factor*normal->z);
        uv->y = 0.5f + factor*normal->y;
    } else if (normal->z > +NORMAL_THRESHOLD) { // front:
        factor = 0.5f / normal->z;
        uv->x = 0.5f + factor*normal->x;
        uv->y = 0.5f + factor*normal->y;
    } else if (normal->z < -NORMAL_THRESHOLD) { // back:
        factor = -0.5f / normal->z;
        uv->x = 1 - (0.5f + factor*normal->x);
        uv->y = 0.5f + factor*normal->y;
    } else if (normal->y > +NORMAL_THRESHOLD) { // top:
        factor = 0.5f / normal->y;
        uv->x = 0.5f + factor*normal->x;
        uv->y = 0.5f + factor*normal->z;
    } else if (normal->y < -NORMAL_THRESHOLD) { // bottom:
        factor = -0.5f / normal->y;
        uv->x = 0.5f + factor*normal->x;
        uv->y = 1 - (0.5f + factor*normal->z);
    }
}

bool hitSpheresSimple(RayHit* closest_hit, bool skip_out_of_view, u8 mask) {
    f32 t, dt, r, closest_distance = closest_hit->distance;
    vec3 _I, *I = &_I, _C, *C = &_C;
    bool found = false;

    f32 distance;
    vec3 *Rd = &closest_hit->ray_direction,
         *Ro = &closest_hit->ray_origin,
         *P = &closest_hit->position,
         *N = &closest_hit->normal;

    // Loop over all the spheres and intersect the ray against them:
    Sphere* hit_sphere = scene.spheres;
    Sphere* last_sphere = scene.spheres + scene.sphere_count;
    u8 sphere_id = 1;
    for (Sphere* sphere = scene.spheres; sphere != last_sphere; sphere++, sphere_id <<= 1) {
        if (skip_out_of_view && !(sphere_id & mask))
            continue;

        subVec3(&sphere->position, Ro, C);
        t = dotVec3(C, Rd);
        if (t > 0) {
            scaleVec3(Rd, t, I);
            isubVec3(I, C);
            r = sphere->radius;
            dt = r*r - squaredLengthVec3(I);
            if (dt > 0 && t*t > dt*dt) { // Inside the sphere
                distance = t - sqrtf(dt);
                if (distance > 0 && distance < closest_distance) {
                    closest_distance = distance;
                    hit_sphere = sphere;
                    found = true;
                }
            }
        }
    }

    if (found) {
        closest_hit->distance = closest_distance;
        C = &hit_sphere->position;
        r = hit_sphere->radius;

        scaleVec3(Rd, closest_distance, P);
        iaddVec3(P, Ro);
        subVec3(P, C, N);
        if (r != 1) iscaleVec3(N, 1/r);

        closest_hit->material_id = hit_sphere->material_id;
    }

    return found;
}

inline void populateHit(RayHit *hit, Sphere *sphere, Material *materials, f32 distance, bool get_uv) {
    getSphereHitPosition(&hit->ray_origin, &hit->ray_direction, distance, &hit->position);
    getSphereHitNormal(&hit->position, &sphere->position, &hit->normal);
    if (get_uv && materials[sphere->material_id].uses & TRANSPARENCY) {
        vec3 rotated_normal;
        mulVec3Mat3(&hit->normal, &sphere->rotation_matrix, &rotated_normal);
        getSphereHitUV(&rotated_normal, &hit->uv);
    }
    hit->distance = distance;
}
bool hitSpheres(Sphere* spheres, u8 sphere_count, Material *materials, RayHit* closest_hit, bool skip_out_of_view, u8 mask) {
    f32 t, dt, r, d, closest_distance = closest_hit->distance;
    vec3 _I, *I = &_I, _C, *C = &_C;
    bool found = false;
    u8 sphere_id;
    Sphere *sphere = spheres;
    SphereHit *sphere_hit = sphere_hits;

    // Loop over all the spheres and intersect the ray against them:
    for (sphere_id = 0; sphere_id != sphere_count; sphere_id++, sphere++, sphere_hit++) {
        sphere_hit->hit = false;

        if (skip_out_of_view && !(sphere_id & mask))
            continue;

        subVec3(&sphere->position, &closest_hit->ray_origin, C);
        t = dotVec3(C, &closest_hit->ray_direction);
        if (t > 0) {
            scaleVec3(&closest_hit->ray_direction, t, I);
            isubVec3(I, C);
            r = sphere->radius;
            dt = r*r - squaredLengthVec3(I);
            if (dt > 0) { // Inside the sphere
                d = sqrtf(dt);
                sphere_hit->hit = true;
                sphere_hit->inner_hit_distance = t + d;
                sphere_hit->outer_hit_distance = t - d;
                found = true;
            }
        }
    }

    if (found) {
        bool inner, outer, populate_hit = true;

        sphere = scene.spheres;
        sphere_hit = sphere_hits;
        Sphere* hit_sphere = sphere;
        for (sphere_id = 0; sphere_id != SPHERE_COUNT; sphere_id++, sphere++, sphere_hit++) {
            if (!sphere_hit->hit) continue;
            inner = sphere_hit->inner_hit_distance > 0 && sphere_hit->inner_hit_distance < closest_distance;
            outer = sphere_hit->outer_hit_distance > 0 && sphere_hit->outer_hit_distance < closest_distance;
            if (inner || outer) {
                populate_hit = true;

                    if (materials[sphere->material_id].uses & TRANSPARENCY) {
                        if (outer) {
                            populateHit(closest_hit, sphere, materials, sphere_hit->outer_hit_distance, true);
                            if (((u8)(closest_hit->uv.x * 8) % 2) &&
                                (((u8)((closest_hit->uv.y + 0.125) * 8)) % 2)) {
                                if (inner) {
                                    populateHit(closest_hit, sphere, materials, sphere_hit->inner_hit_distance, true);
                                    if (((u8)(closest_hit->uv.x * 8) % 2) &&
                                        (((u8)((closest_hit->uv.y + 0.125) * 8)) % 2))
                                        continue;
                                }
                            }
                        } else {
                            populateHit(closest_hit, sphere, materials, sphere_hit->inner_hit_distance, true);
                            if (((u8)(closest_hit->uv.x * 8) % 2) &&
                                (((u8)((closest_hit->uv.y + 0.125) * 8)) % 2))
                                continue;
                        }

                        populate_hit = false;
                    }

                hit_sphere = sphere;
                closest_distance = outer ? sphere_hit->outer_hit_distance : sphere_hit->inner_hit_distance;
            }
        }

        if (populate_hit)
            populateHit(closest_hit, hit_sphere, materials, closest_distance, true);

        closest_hit->material_id = hit_sphere->material_id;
    }

    return found;
}

inline bool hitPlane(vec3 *position, vec3 *normal, vec3 *ray_direction, vec3 *ray_origin, f32 *hit_distance) {
    f32 Rd_dot_n = dotVec3(ray_direction, normal);
    if (Rd_dot_n >= 0 ||
        -Rd_dot_n < EPS)
        return false;

    vec3 ray_origin_to_position;
    subVec3(position, ray_origin, &ray_origin_to_position);
    f32 p_dot_n = dotVec3(&ray_origin_to_position, normal);
    if (p_dot_n >= 0 ||
        -p_dot_n < EPS)
        return false;

    *hit_distance = p_dot_n / Rd_dot_n;
    return true;
}
bool hitPlanes(RayHit* closest_hit) {
    vec3 *Rd = &closest_hit->ray_direction;
    vec3 *Ro = &closest_hit->ray_origin;
    f32 closest_distance = closest_hit->distance;
    f32 hit_distance = 0;
    bool found = false;

    // Loop over all the planes and intersect the ray against them:
    Plane* hit_plane = scene.planes;
    Plane* last_plane = scene.planes + scene.plane_count;
    for (Plane* plane = scene.planes; plane != last_plane; plane++) {
        if (!hitPlane(&plane->position, &plane->normal, Rd, Ro, &hit_distance))
            continue;

        if (hit_distance < closest_distance) {
            closest_distance = hit_distance;
            hit_plane = plane;
            found = true;
        }
    }

    if (found) {
        scaleVec3(Rd, closest_distance, &closest_hit->position);
        iaddVec3(&closest_hit->position, Ro);
        closest_hit->normal = hit_plane->normal;
        closest_hit->distance = closest_distance;
        closest_hit->inner = false;
        closest_hit->material_id = hit_plane->material_id;
    }

    return found;
}

bool hitCubes(RayHit* closest_hit) {
    vec3 *Rd = &closest_hit->ray_direction;
    vec3 *Ro = &closest_hit->ray_origin;
    vec3 hit_position, closest_hit_position, closest_hit_normal;
    f32 hit_distance, closest_hit_distance = closest_hit->distance;
    u8 closest_hit_material = 0;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Cube* last_cube = scene.cubes + scene.cube_count;
    for (Cube* cube = scene.cubes; cube != last_cube; cube++) {
        Triangle* last_triangle = cube->triangles + 12;
        for (Triangle* triangle = cube->triangles; triangle != last_triangle; triangle++) {
            if (hitPlane(triangle->p1, triangle->normal, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                vec3 p1p2; subVec3(triangle->p2, triangle->p1, &p1p2);
                vec3 p2p3; subVec3(triangle->p3, triangle->p2, &p2p3);
                vec3 p3p1; subVec3(triangle->p1, triangle->p3, &p3p1);

                vec3 p1P; subVec3(&hit_position, triangle->p1, &p1P);
                vec3 p2P; subVec3(&hit_position, triangle->p2, &p2P);
                vec3 p3P; subVec3(&hit_position, triangle->p3, &p3P);

                vec3 c1; crossVec3(&p1P, &p1p2, &c1);
                vec3 c2; crossVec3(&p2P, &p2p3, &c2);
                vec3 c3; crossVec3(&p3P, &p3p1, &c3);

                if (dotVec3(triangle->normal, &c1) > 0 &&
                    dotVec3(triangle->normal, &c2) > 0 &&
                    dotVec3(triangle->normal, &c3) > 0) {
                    closest_hit_distance = hit_distance;
                    closest_hit_position = hit_position;
                    closest_hit_normal   = *triangle->normal;
                    closest_hit_material = cube->material_id;

                    found = true;
                }
            }
        }
    }

    if (found) {
        closest_hit->normal = closest_hit_normal;
        closest_hit->position = closest_hit_position;
        closest_hit->distance = closest_hit_distance;
        closest_hit->material_id = closest_hit_material;
    }

    return found;
}

// ad - bc > 0
// a = p3.x = 1/2
// b = p3.y = s3/2
// c = P.x
// d = P.y
//
// P.y > s3*P.x

// ad - bc > 0
// a = P.x
// b = P.y
// c = p2.x = 1
// d = P2.y = 0
//
// 0 < P.y

// ad - bc > 0
// a = P.x - p2.x = P.x - 1 = (P.x - 1)
// b = P.y - p2.y = P.y - 0 = P.y
// c = p3.x - p2.x = 1/2 - 1 = -1/2
// d = P3.y - p2.y = s3/2 - 0 = s3/2
//
// (P.x - 1)s3/2 > P.y*-1/2
// (P.x - 1)s3 > P.y*-1
// -1*(1 - P.x)s3 > -1*P.y
// (1 - P.x)s3 < P.y

bool hitImplicitTetrahedra(RayHit* closest_hit) {
    vec3 *Rd = &closest_hit->ray_direction;
    vec3 *Ro = &closest_hit->ray_origin;
    vec3 hit_position, closest_hit_position, hit_position_tangent, closest_hit_normal;
    f32 x, y, hit_distance, closest_hit_distance = closest_hit->distance;
    u8 closest_hit_material = 0;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* last_tetrahedron = scene.tetrahedra + scene.tetrahedron_count;
    for (Tetrahedron* tetrahedron = scene.tetrahedra; tetrahedron != last_tetrahedron; tetrahedron++) {
        Triangle* last_triangle = tetrahedron->triangles + 4;
        for (Triangle* triangle = tetrahedron->triangles; triangle != last_triangle; triangle++) {
            if (hitPlane(triangle->p1, triangle->normal, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

//                vec3 p1p2; subVec3(triangle->p2, triangle->p1, &p1p2);
//                vec3 p2p3; subVec3(triangle->p3, triangle->p2, &p2p3);
//                vec3 p3p1; subVec3(triangle->p1, triangle->p3, &p3p1);
//
//                vec3 p1P; subVec3(&hit_position, triangle->p1, &p1P);
//                vec3 p2P; subVec3(&hit_position, triangle->p2, &p2P);
//                vec3 p3P; subVec3(&hit_position, triangle->p3, &p3P);
//
//                vec3 c1; crossVec3(&p1P, &p1p2, &c1);
//                vec3 c2; crossVec3(&p2P, &p2p3, &c2);
//                vec3 c3; crossVec3(&p3P, &p3p1, &c3);
//
//                if (dotVec3(triangle->normal, &c1) > 0 &&
//                    dotVec3(triangle->normal, &c2) > 0 &&
//                    dotVec3(triangle->normal, &c3) > 0) {
//                    closest_hit_distance = hit_distance;
//                    closest_hit_position = hit_position;
//                    closest_hit_normal   = *triangle->normal;
//                    closest_hit_material = tetrahedron->material;
//
//                    found = true;
//                }

                subVec3(&hit_position, triangle->p1, &hit_position_tangent);
                imulVec3Mat3(&hit_position_tangent, &triangle->world_to_tangent);
                x = hit_position_tangent.x;
                y = hit_position_tangent.y;

                if (y > 0 && y < x*SQRT3 && y < (1 - x)*SQRT3) {
                    closest_hit_distance = hit_distance;
                    closest_hit_position = hit_position;
                    closest_hit_normal   = *triangle->normal;
                    closest_hit_material = tetrahedron->material_id;

                    found = true;
                }
            }
        }
    }

    if (found) {
        closest_hit->normal = closest_hit_normal;
        closest_hit->position = closest_hit_position;
        closest_hit->distance = closest_hit_distance;
        closest_hit->material_id = closest_hit_material;
    }

    return found;
}

bool hitTetrahedra(RayHit* closest_hit) {
    vec3 *Rd = &closest_hit->ray_direction;
    vec3 *Ro = &closest_hit->ray_origin;
    vec3 hit_position, closest_hit_position, closest_hit_normal;
    f32 hit_distance, closest_hit_distance = closest_hit->distance;
    u8 closest_hit_material = 0;
    bool found = false;

    // Loop over all tetrahedra and intersect the ray against them:
    Tetrahedron* last_tetrahedron = scene.tetrahedra + scene.tetrahedron_count;
    for (Tetrahedron* tetrahedron = scene.tetrahedra; tetrahedron != last_tetrahedron; tetrahedron++) {
        Triangle* last_triangle = tetrahedron->triangles + 4;
        for (Triangle* triangle = tetrahedron->triangles; triangle != last_triangle; triangle++) {
            if (hitPlane(triangle->p1, triangle->normal, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {

                scaleVec3(Rd, hit_distance, &hit_position);
                iaddVec3(&hit_position, Ro);

                vec3 p1p2; subVec3(triangle->p2, triangle->p1, &p1p2);
                vec3 p2p3; subVec3(triangle->p3, triangle->p2, &p2p3);
                vec3 p3p1; subVec3(triangle->p1, triangle->p3, &p3p1);

                vec3 p1P; subVec3(&hit_position, triangle->p1, &p1P);
                vec3 p2P; subVec3(&hit_position, triangle->p2, &p2P);
                vec3 p3P; subVec3(&hit_position, triangle->p3, &p3P);

                vec3 c1; crossVec3(&p1P, &p1p2, &c1);
                vec3 c2; crossVec3(&p2P, &p2p3, &c2);
                vec3 c3; crossVec3(&p3P, &p3p1, &c3);

                if (dotVec3(triangle->normal, &c1) > 0 &&
                    dotVec3(triangle->normal, &c2) > 0 &&
                    dotVec3(triangle->normal, &c3) > 0) {
                    closest_hit_distance = hit_distance;
                    closest_hit_position = hit_position;
                    closest_hit_normal   = *triangle->normal;
                    closest_hit_material = tetrahedron->material_id;

                    found = true;
                }
            }
        }
    }

    if (found) {
        closest_hit->normal = closest_hit_normal;
        closest_hit->position = closest_hit_position;
        closest_hit->distance = closest_hit_distance;
        closest_hit->material_id = closest_hit_material;
    }

    return found;
}

//bool hitTriangles(RayHit* closest_hit, Material** material_ptr) {
//    vec3 *Rd = &closest_hit->ray_direction;
//    vec3 *Ro = &closest_hit->ray_origin;
//    vec3 hit_position, closest_hit_position, closest_hit_normal;
//    f32 hit_distance, closest_hit_distance = closest_hit->distance;
//    Material *closest_hit_material = NULL;
//    bool found = false;
//
//    // Loop over all tetrahedra and intersect the ray against them:
//    Triangle* last_triangle = scene.triangles + scene.triangle_count;
//    for (Triangle* triangle = scene.triangles; triangle != last_triangle; triangle++) {
//        if (hitPlane(triangle->p1, triangle->normal, Rd, Ro, &hit_distance) && hit_distance < closest_hit_distance) {
//
//            scaleVec3(Rd, hit_distance, &hit_position);
//            iaddVec3(&hit_position, Ro);
//
//            vec3 p1p2; subVec3(triangle->p2, triangle->p1, &p1p2);
//            vec3 p2p3; subVec3(triangle->p3, triangle->p2, &p2p3);
//            vec3 p3p1; subVec3(triangle->p1, triangle->p3, &p3p1);
//
//            vec3 p1P; subVec3(&hit_position, triangle->p1, &p1P);
//            vec3 p2P; subVec3(&hit_position, triangle->p2, &p2P);
//            vec3 p3P; subVec3(&hit_position, triangle->p3, &p3P);
//
//            vec3 c1; crossVec3(&p1P, &p1p2, &c1);
//            vec3 c2; crossVec3(&p2P, &p2p3, &c2);
//            vec3 c3; crossVec3(&p3P, &p3p1, &c3);
//
//            if (dotVec3(triangle->normal, &c1) > 0 &&
//                dotVec3(triangle->normal, &c2) > 0 &&
//                dotVec3(triangle->normal, &c3) > 0) {
//                closest_hit_distance = hit_distance;
//                closest_hit_position = hit_position;
//                closest_hit_normal   = *triangle->normal;
////                closest_hit_material = tetrahedron->material;
//
//                found = true;
//            }
//        }
//    }
//
//    if (found) {
//        closest_hit->normal = closest_hit_normal;
//        closest_hit->position = closest_hit_position;
//        closest_hit->distance = closest_hit_distance;
//        *material_ptr = closest_hit_material;
//    }
//
//    return found;
//}