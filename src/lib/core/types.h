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

typedef float  f32;
typedef double f64;


typedef union Color {
    struct {
        u8 B, G, R, Z;
    };
    u32 value;
} Color;