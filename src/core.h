#pragma once
#include "linalg.h"

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

Memory memory = {
    MEMORY_BASE,
    MEMORY_SIZE
};

typedef struct Mouse {
    Vector2 current_position;
    Vector2 prior_position;
} Mouse;

typedef struct Camera {
    f32 focal_length;
    Matrix3x3 matrix;
    Vector3* position;
} Camera;

App app = {0, 0};
Keyboard keyboard = {0};
Mouse mouse = {{0.0f, 0.0f}, {-1.0f, -1.0f}};
Camera camera = {1.0f};
Pixel* pixel;

FrameBuffer frame_buffer = {
    INITIAL_WIDTH,
    INITIAL_HEIGHT,
    INITIAL_WIDTH * PIXEL_SIZE,
    INITIAL_WIDTH * INITIAL_HEIGHT
};

f32 movement_speed = 0.5f;
Vector3* up_direction = &yaw_matrix.j;
Vector3* right_direction = &yaw_matrix.i;
Vector3* forward_direction = &yaw_matrix.k;

Vector3* up;
Vector3* right;
Vector3* forward;

void* allocate_memory(u64 size) {
    memory.occupied += size;
    //if (memory.occupied > memory.size)
    //    abort();

    void* address = memory.address;
    memory.address += size;
    return address;
}

void init_core() {
    setMatrixToIdentity(&camera.matrix);
    up = (Vector3*)allocate_memory(sizeof(Vector3));
    right = (Vector3*)allocate_memory(sizeof(Vector3));
    forward = (Vector3*)allocate_memory(sizeof(Vector3));
    
    frame_buffer.pixels = (Pixel*)allocate_memory(RENDER_SIZE);
}

void rotate_camera() {
    rotate(
        &camera.matrix,
        (mouse.current_position.x - mouse.prior_position.x) / -1000.0f,
        (mouse.current_position.y - mouse.prior_position.y) / -1000.0f,
        0.0f
    );
}

void update() {
    if (keyboard.pressed) {
        if (keyboard.pressed & FORWARD ||
            keyboard.pressed & BACKWARD) {
            scale(forward_direction, keyboard.pressed & FORWARD ? movement_speed : -movement_speed, forward);
            iadd(camera.position, forward);
        }
        if (keyboard.pressed & RIGHT ||
            keyboard.pressed & LEFT) {
            scale(right_direction, keyboard.pressed & RIGHT ? movement_speed : -movement_speed, right);            
            iadd(camera.position, right);
        }
        if (keyboard.pressed & UP ||
            keyboard.pressed & DOWN) {
            scale(up_direction, keyboard.pressed & UP ? movement_speed : -movement_speed, up);            
            iadd(camera.position, up);
        }
    }
}

void print_to_string(char* message, u32 number, char* string) {
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

void print_title_to_string(u16 width, u16 height, u16 fps, char* title, char* title_string) {
    title_string[31] = '\0';
    title_string[30] = 'S';
    title_string[29] = 'P';
    title_string[28] = 'F';

    char* ptr = title_string + 27;
    
    int temp;

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


//#define RENDER(name) void name(int pixel_count, u32 pixels[])
//typedef RENDER(render);