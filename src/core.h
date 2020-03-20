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
static FrameBuffer frame_buffer;
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

typedef struct TextLine {
    char* string;
    u8 length, y, n1, n2;
} TextLine;

static TextLine RESOLUTION;
static TextLine FRAME_RATE;
static TextLine FRAME_TIME;
static TextLine NAVIGATION;

void init_core() {
    frame_buffer.pixels = (u32*)allocate_memory(RENDER_SIZE);

    RESOLUTION.string = (char*)allocate_memory(OVR_WIDTH);
    FRAME_RATE.string = (char*)allocate_memory(OVR_WIDTH);
    FRAME_TIME.string = (char*)allocate_memory(OVR_WIDTH);
    NAVIGATION.string = (char*)allocate_memory(OVR_WIDTH);

    RESOLUTION.string[0] = 'R';
    RESOLUTION.string[1] = 'E';
    RESOLUTION.string[2] = 'S';
    RESOLUTION.string[3] = ':';
    RESOLUTION.string[4] = ' ';
    RESOLUTION.string[5] = '9';
    RESOLUTION.string[6] = '9';
    RESOLUTION.string[7] = '9';
    RESOLUTION.string[8] = '9';
    RESOLUTION.string[9] = 'x';
    RESOLUTION.string[10] = '9';
    RESOLUTION.string[11] = '9';
    RESOLUTION.string[12] = '9';
    RESOLUTION.string[13] = '9';
    RESOLUTION.string[14] = '\0';

    FRAME_RATE.string[0] = 'F';
    FRAME_RATE.string[1] = 'P';
    FRAME_RATE.string[2] = 'S';
    FRAME_RATE.string[3] = ':';
    FRAME_RATE.string[4] = ' ';
    FRAME_RATE.string[5] = '9';
    FRAME_RATE.string[6] = '9';
    FRAME_RATE.string[7] = '9';
    FRAME_RATE.string[8] = '9';
    FRAME_RATE.string[9] = '\0';

    FRAME_TIME.string[0] = 'F';
    FRAME_TIME.string[1] = 'M';
    FRAME_TIME.string[2] = 'S';
    FRAME_TIME.string[3] = ':';
    FRAME_TIME.string[4] = ' ';
    FRAME_TIME.string[5] = '9';
    FRAME_TIME.string[6] = '9';
    FRAME_TIME.string[7] = '9';
    FRAME_TIME.string[8] = '9';
    FRAME_TIME.string[9] = '\0';

    NAVIGATION.string[0] = 'N';
    NAVIGATION.string[1] = 'A';
    NAVIGATION.string[2] = 'V';
    NAVIGATION.string[3] = ':';
    NAVIGATION.string[4] = ' ';
    NAVIGATION.string[5] = 'O';
    NAVIGATION.string[6] = 'r';
    NAVIGATION.string[7] = 'b';
    NAVIGATION.string[8] = '\0';

    RESOLUTION.length = (u8)strlen(RESOLUTION.string);
    FRAME_RATE.length = (u8)strlen(FRAME_RATE.string);
    FRAME_TIME.length = (u8)strlen(FRAME_TIME.string);
    NAVIGATION.length = (u8)strlen(NAVIGATION.string);

    RESOLUTION.y = 10;
    FRAME_RATE.y = 30;
    FRAME_TIME.y = 50;
    NAVIGATION.y = 70;

    RESOLUTION.n1 = 8;
    RESOLUTION.n2 = 13;
    FRAME_RATE.n1 = 8;
    FRAME_TIME.n1 = 8;
}

void onMouseCaptured() {
    mouse.is_captured = true;
    NAVIGATION.string[5] = 'F';
    NAVIGATION.string[6] = 'p';
    NAVIGATION.string[7] = 's';
}
void onMouseUnCaptured() {
    mouse.is_captured = false;
    NAVIGATION.string[5] = 'O';
    NAVIGATION.string[6] = 'r';
    NAVIGATION.string[7] = 'b';
}