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

#define Kilobytes(value) ((value)*1024LL)
#define Megabytes(value) (Kilobytes(value)*1024LL)
#define Gigabytes(value) (Megabytes(value)*1024LL)
#define Terabytes(value) (Gigabytes(value)*1024LL)

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))

static u8 FORWARD = 1;
static u8 BACKWARD = 2;
static u8 LEFT = 4;
static u8 RIGHT = 8;
static u8 UP = 16;
static u8 DOWN = 32;
static u8 QUIT = 64;
typedef u8 Keyboard;

inline char* fill_string(u32 number, char* ptr) {
    u32 number_div_10;
    if (number)
        do {
            number_div_10 = number / 10;
            *ptr-- = '0' + number - number_div_10;
            number = number_div_10;
        } while (number);
    else
        *ptr-- = '0';

    return ptr;
}

inline void print_numbers_to_string(u32 number_1, u32 number_2, char* string) {
    char* ptr = string + 13;
    string[0] = ' ';
    string[14] = '\n';
    string[15] = '\0';

    ptr = fill_string(number_1, ptr);
    *ptr-- = ' ';
    ptr = fill_string(number_2, ptr);
    while (ptr > string) *ptr-- = ' ';
}

//#define RENDER(name) void name(int pixel_count, u32 pixels[])
//typedef RENDER(render);