#pragma once
#include "types.h"
#include "math2D.h"

#define TRUE 1
#define FALSE 0
#define INT_MAX	2147483647
#define flr(x) (x < (u32)x ? (u32)x - 1 : (u32)x)

#define Kilobytes(value) ((value)*1024LL)
#define Megabytes(value) (Kilobytes(value)*1024LL)
#define Gigabytes(value) (Megabytes(value)*1024LL)
#define Terabytes(value) (Gigabytes(value)*1024LL)
#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

#define INITIAL_WIDTH 800
#define INITIAL_HEIGHT 600
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
    u8 is_active, is_running;
} App;
App app = {0, 1};

typedef struct Keyboard {
    u8 pressed;
} Keyboard;
Keyboard keyboard = {0};

u8 FORWARD  = 1;
u8 BACKWARD = 2;
u8 LEFT     = 4;
u8 RIGHT    = 8;
u8 UP       = 16;
u8 DOWN     = 32;

typedef union Pixel {
    u32 color;
    struct {
        u8 R, G, B, A;
    };
} Pixel;
Pixel* pixel;

typedef struct FrameBuffer {
    u16 width, height;
    u32 size;
    Pixel* pixels;

} FrameBuffer;
FrameBuffer frame_buffer = {
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    INITIAL_WIDTH * INITIAL_HEIGHT
};

void* allocate_memory(u64 size) {
    memory.occupied += size;
    //if (memory.occupied > memory.size)
    //    abort();

    void* address = memory.address;
    memory.address += size;
    return address;
}

void printIntoString(char* message, u32 number, char* string) {
    char* ptr = string + 13;
    string[0] = ' ';
    string[14] = '\n';
    string[15] = '\0';

    int temp;

    if (number)
        do {
            temp = number;
            number /= 10;
            *ptr-- = (char)('0' + temp - number * 10);
        } while (number);
    else
        *ptr-- = '0';

    while (ptr > string)
        *ptr-- = ' ';

    while (*message != '\0')
        *ptr++ = *message++;
}

void printTitleIntoString(u16 width, u16 height, u16 fps, char* title, char* title_string) {
    title_string[31] = '\0';
    title_string[30] = 'S';
    title_string[29] = 'P';
    title_string[28] = 'F';

    char* ptr = title_string + 13;
    
    u16 temp;

    do {
        temp = fps;
        fps /= 10;
        *ptr-- = (char)('0' + temp - fps * 10);
    } while (fps);

    *ptr-- = ' ';

    do {
        temp = height;
        height /= 10;
        *ptr-- = (char)('0' + temp - height * 10);
    } while (height);

    *ptr-- = 'x';
    
    do {
        temp = width;
        width /= 10;
        *ptr-- = (char)('0' + temp - width * 10);
    } while (width);

    while (ptr > title_string)
        *ptr-- = ' ';

    while (*title != '\0')
        *ptr++ = *title++;
}

void init_core() {
    frame_buffer.pixels = (Pixel*)allocate_memory(RENDER_SIZE);
}