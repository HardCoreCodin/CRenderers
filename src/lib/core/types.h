#pragma once
#pragma warning(disable : 4201)

#ifndef __cplusplus
#define false 0
#define true 1
typedef unsigned char      bool;
#endif

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

typedef signed   short     i16;
typedef signed   int       i32;

typedef float  f32;
typedef double f64;

typedef unsigned char byte;

typedef void (*CallBack)();

// Math:
// ====
#define EPS 0.0001f
#define SQRT3 1.73205080757f
#define SQRT_OF_TWO_THIRDS 0.81649658092f
#define SQRT_OF_THREE_OVER_SIX 0.28867513459f
#define SQRT_OF_THREE_OVER_THREE 0.57735026919f

typedef struct { i32 x, y;    } vec2i;
typedef struct { f32 x, y;    } vec2;
typedef struct { f32 x, y, z; } vec3;
typedef struct { vec2 X, Y;    } mat2;
typedef struct { vec3 X, Y, Z; } mat3;

// Core:
// ====
#define MAX_COLOR_VALUE 0xFF
typedef struct { u8 B, G, R, A; } Color;
typedef struct { u16 min, max; } range2i;
typedef struct { vec3 min, max; } AABB;
typedef struct { range2i x_range, y_range; } Bounds2Di;

typedef union {
    Color color;
    u32 value;
} Pixel;
#define setPixelColor(pixel, color) \
        pixel->color.R = color.x > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.x; \
        pixel->color.G = color.y > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.y; \
        pixel->color.B = color.z > MAX_COLOR_VALUE ? MAX_COLOR_VALUE : (u8)color.z

typedef struct {
    mat2 matrix,
         rotation_matrix,
         rotation_matrix_inverted;
    vec2 position,
         *right_direction,
         *forward_direction;
} xform2;

typedef struct {
    mat3 matrix,
         yaw_matrix,
         pitch_matrix,
         roll_matrix,
         rotation_matrix,
         rotation_matrix_inverted;

    vec3 position,
         *up_direction,
         *right_direction,
         *forward_direction;
} xform3;

#ifdef __CUDACC__
    #define initKernel(count, width) \
        u32 i = blockDim.x * blockIdx.x + threadIdx.x; \
        if (i >= count) return; \
        f32 x = i % width; \
        f32 y = i / width

    #define setupKernel(count) \
        u16 threads = CUDA_MAX_THREADS; \
        u16 blocks  = count / CUDA_MAX_THREADS; \
        if (count < CUDA_MAX_THREADS) { \
            threads = count; \
            blocks = 1; \
        }

    #ifndef NDEBUG
        #define gpuErrchk(ans) ans
    #else
        #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

        #include <stdio.h>
        #include <stdlib.h>

        inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
            if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
            }
        }
    #endif
#endif