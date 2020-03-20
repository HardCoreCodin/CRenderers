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

#define INITIAL_WIDTH 800
#define INITIAL_HEIGHT 600
#define PIXEL_SIZE 4

#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)
#define MEMORY_SIZE Gigabytes(1)
#define MEMORY_BASE Terabytes(2)

#define OVR_LEFT 10
#define OVR_TOP 10
#define OVR_WIDTH 120
#define OVR_HEIGHT 100


typedef struct Memory {
    u64 base, size, occupied;
    u8* address;
} Memory;
Memory memory = {
    MEMORY_BASE,
    MEMORY_SIZE
};

typedef struct App {
    u8 is_running;
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
FrameBuffer frame_buffer = {
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    INITIAL_WIDTH * INITIAL_HEIGHT
};
u32* pixel;

void* allocate_memory(u64 size) {
    memory.occupied += size;
    //if (memory.occupied > memory.size)
    //    abort();

    void* address = memory.address;
    memory.address += size;
    return address;
}

void printNumberIntoString(u16 number, char* str) {
    char *string = str;
    if (number) {
        u16 temp;
        temp = number;
        number /= 10;
        *string-- = (char)('0' + temp - number * 10);

        if (number) {
            temp = number;
            number /= 10;
            *string-- = (char)('0' + temp - number * 10);

            if (number) {
                temp = number;
                number /= 10;
                *string-- = (char)('0' + temp - number * 10);

                if (number) {
                    temp = number;
                    number /= 10;
                    *string = (char)('0' + temp - number * 10);
                } else
                    *string = ' ';
            } else {
                *string-- = ' ';
                *string-- = ' ';
            }
        } else {
            *string-- = ' ';
            *string-- = ' ';
            *string-- = ' ';
        }
    } else {
        *string-- = '0';
        *string-- = ' ';
        *string-- = ' ';
        *string   = ' ';
    }
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

inline f32 approach(f32 from, f32 to, f32 step) {
    f32 delta = to - from;
    if (delta > step) return from + step;
    if (delta < -step) return from - step;
    return to;
}

static char* RESOLUTION_STRING = "RES: 9999x9999";
static char* FRAME_RATE_STRING = "FPS: 9999";
static char* FRAME_TIME_STRING = "FMS: 9999";
static char* NAVIGATION_STRING = "NAV: Orb";

static u8 RESOLUTION_STRING_LENGTH;
static u8 FRAME_RATE_STRING_LENGTH;
static u8 FRAME_TIME_STRING_LENGTH;
static u8 NAVIGATION_STRING_LENGTH;

static u8 RESOLUTION_STRING_START;
static u8 FRAME_RATE_STRING_START;
static u8 FRAME_TIME_STRING_START;
static u8 NAVIGATION_STRING_START;

void init_core() {
    RESOLUTION_STRING = (char*)allocate_memory(OVR_WIDTH);
    FRAME_RATE_STRING = (char*)allocate_memory(OVR_WIDTH);
    FRAME_TIME_STRING = (char*)allocate_memory(OVR_WIDTH);
    NAVIGATION_STRING = (char*)allocate_memory(OVR_WIDTH);

    RESOLUTION_STRING = "RES: 9999x9999";
    FRAME_RATE_STRING = "FPS: 9999";
    FRAME_TIME_STRING = "FMS: 9999";
    NAVIGATION_STRING = "NAV: Orb";

    frame_buffer.pixels = (u32*)allocate_memory(RENDER_SIZE);

    RESOLUTION_STRING_LENGTH = (u8)strlen(RESOLUTION_STRING);
    FRAME_RATE_STRING_LENGTH = (u8)strlen(FRAME_RATE_STRING);
    FRAME_TIME_STRING_LENGTH = (u8)strlen(FRAME_TIME_STRING);
    NAVIGATION_STRING_LENGTH = (u8)strlen(NAVIGATION_STRING);

    RESOLUTION_STRING_START = 10;
    FRAME_RATE_STRING_START = 30;
    FRAME_TIME_STRING_START = 50;
    NAVIGATION_STRING_START = 70;
}

void onMouseCaptured() {
    mouse.is_captured = true;
    NAVIGATION_STRING[5] = 'F';
    NAVIGATION_STRING[6] = 'p';
    NAVIGATION_STRING[7] = 's';
}
void onMouseUnCaptured() {
    mouse.is_captured = false;
    NAVIGATION_STRING[5] = 'O';
    NAVIGATION_STRING[6] = 'r';
    NAVIGATION_STRING[7] = 'b';
}