#pragma once

typedef signed char i8;
typedef short       i16;
typedef int         i32;
typedef long long   i64;

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

typedef float  f32;
typedef double f64;

enum Keys {
    FORWARD = 1,
    BACKWARD = 2,
    LEFT = 4,
    RIGHT = 8,
    UP = 16,
    DOWN = 32,
    QUIT = 64
};


typedef struct Memory {
    u64 base, size, occupied;
    u8* address;
} Memory;

typedef struct Pixel {
    u8 R, G, B, A;
} Pixel;

typedef struct FrameBuffer {
    u16 width, height;
    u32 pitch, size;
    Pixel* pixels;
} FrameBuffer;

typedef struct Keyboard {
    u8 pressed;
} Keyboard;

typedef struct App {
    u8 is_active, should_quit;
} App;

//typedef struct Quaternion {Vector3 v; f32 w;} Quaternion;