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

#if defined(__CUDACC__) // NVCC
    #define _align(n) __align__(n)
#elif defined(__GNUC__) // GCC
    #define _align(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER) // MSVC
    #define _align(n) __declspec(align(n))
#else
    #error "Please provide a definition for _align macro for your host compiler!"
#endif


// Math:
// ====
#define EPS 0.0001f
#define SQRT2 1.41421356237f
#define SQRT3 1.73205080757f

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
        color.x *= 255;    \
        color.y *= 255;    \
        color.z *= 255;    \
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
    #define initKernel() \
        u32 i = blockDim.x * blockIdx.x + threadIdx.x; \
        if (i >= d_dimentions->width_times_height) return; \
        f32 x = i % d_dimentions->width; \
        f32 y = i / d_dimentions->width

    #define setupKernel() \
        u16 threads = 256; \
        u16 blocks  = frame_buffer.dimentions.width_times_height / threads; \
        if (frame_buffer.dimentions.width_times_height < threads) { \
            threads = frame_buffer.dimentions.width_times_height; \
            blocks = 1; \
        } \
        if (frame_buffer.dimentions.width_times_height % threads) blocks++;

    #ifndef NDEBUG
        #include <stdio.h>
        #include <stdlib.h>
        #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
        inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
            if (code != cudaSuccess) {
                fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
                if (abort) exit(code);
            }
        }
    #else
        #define gpuErrchk(ans) ans
    #endif
#endif