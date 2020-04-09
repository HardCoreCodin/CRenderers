#pragma once

#ifndef __cplusplus
#define false 0
#define true 1
typedef unsigned char      bool;
#endif

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

typedef signed   char      s8;
typedef signed   short     s16;
typedef signed   int       s32;
typedef signed   long long s64;

typedef float  f32;
typedef double f64;

typedef struct Vector2 {
    f32 x, y;
} Vector2;

typedef struct Vector3 {
    f32 x, y, z;
} Vector3;

typedef struct Color {
    u8 B, G, R;
} Color;

typedef union Pixel {
    struct {
        Color color;
        u8 depth;
    };
    u32 value;
} Pixel;