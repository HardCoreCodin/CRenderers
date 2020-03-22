#pragma once
#include "types.h"
#include "math2D.h"

#define INT_MAX	2147483647
#define flr(x) (x < (u32)x ? (u32)x - 1 : (u32)x)

#define Kilobytes(value) ((value)*1024LL)
#define Megabytes(value) (Kilobytes(value)*1024LL)
#define Gigabytes(value) (Megabytes(value)*1024LL)
#define Terabytes(value) (Gigabytes(value)*1024LL)
#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

#define PIXEL_SIZE 4

#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)
#define MEMORY_SIZE Gigabytes(1)
#define MEMORY_BASE Terabytes(2)

typedef struct Memory {
    u64 base, size, occupied;
    u8* address;
} Memory;
Memory memory = {
    MEMORY_BASE,
    MEMORY_SIZE
};

typedef struct App {
    u8 is_running, is_HUD_visible;
} App;
App app = {1};

typedef struct Keyboard {
    u8 pressed;
} Keyboard;
Keyboard keyboard = {0};

typedef struct Mouse {
    u8 pressed, is_captured;
} Mouse;
Mouse mouse = {0, false};

u8 FORWARD  = 1;
u8 BACKWARD = 2;
u8 LEFT     = 4;
u8 RIGHT    = 8;
u8 UP       = 16;
u8 DOWN     = 32;
u8 MIDDLE   = 64;

typedef union Color {
    struct {
        u8 B, G, R, A;
    };
    u32 value;
} Color;
static Color color;

typedef struct FrameBuffer {
    u16 width, height;
    u32 size;
    u32* pixels;

} FrameBuffer;
static FrameBuffer frame_buffer;
u32* pixel;

void* allocate_memory(u64 size) {
    memory.occupied += size;

    void* address = memory.address;
    memory.address += size;
    return address;
}

inline f32 approach(f32 from, f32 to, f32 step) {
    f32 delta = to - from;
    if (delta > step) return from + step;
    if (delta < -step) return from - step;
    return to;
}

void init_core() {
    frame_buffer.pixels = (u32*)allocate_memory(RENDER_SIZE);
}