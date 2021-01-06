#pragma once
#pragma warning(disable : 4201)

#include "lib/core/types.h"

#define MAX_WIDTH 3840
#define MAX_HEIGHT 2160

#define PIXEL_SIZE 4
#define RENDER_SIZE Megabytes(8 * PIXEL_SIZE)

#define HUD_LENGTH 140

bool is_running = true;
bool use_GPU = true;
bool show_BVH = false;
bool show_SSB = false;

enum RenderMode {
    Normals,
    Beauty,
    Depth,
    UVs
};
enum RenderMode render_mode = Beauty;

#ifdef __CUDACC__
    #define CUDA_MAX_THREADS 1024

    bool last_rendered_on_CPU = true;
#endif

typedef void (*UpdateWindowTitle)();
typedef void (*PrintDebugString)(char* str);

UpdateWindowTitle updateWindowTitle;
PrintDebugString printDebugString;

Color WHITE,
      GREY,
      BLACK,
      RED,
      GREEN,
      BLUE,
      CYAN,
      MAGENTA,
      YELLOW;

void initAppGlobals() {
    BLACK.R = 0;
    BLACK.G = 0;
    BLACK.B = 0;
    BLACK.A = MAX_COLOR_VALUE;

    WHITE.R = MAX_COLOR_VALUE;
    WHITE.G = MAX_COLOR_VALUE;
    WHITE.B = MAX_COLOR_VALUE;
    WHITE.A = MAX_COLOR_VALUE;

    GREY.R = MAX_COLOR_VALUE/2;
    GREY.G = MAX_COLOR_VALUE/2;
    GREY.B = MAX_COLOR_VALUE/2;
    GREY.A = MAX_COLOR_VALUE;

    RED.R = MAX_COLOR_VALUE;
    RED.G = 0;
    RED.B = 0;
    RED.A = MAX_COLOR_VALUE;

    GREEN.R = 0;
    GREEN.G = MAX_COLOR_VALUE;
    GREEN.B = 0;
    GREEN.A = MAX_COLOR_VALUE;

    BLUE.R = 0;
    BLUE.G = 0;
    BLUE.B = MAX_COLOR_VALUE;
    BLUE.A = MAX_COLOR_VALUE;

    CYAN.R = 0;
    CYAN.G = MAX_COLOR_VALUE;
    CYAN.B = MAX_COLOR_VALUE;
    CYAN.A = MAX_COLOR_VALUE;

    MAGENTA.R = MAX_COLOR_VALUE;
    MAGENTA.G = 0;
    MAGENTA.B = MAX_COLOR_VALUE;
    MAGENTA.A = MAX_COLOR_VALUE;

    YELLOW.R = MAX_COLOR_VALUE;
    YELLOW.G = MAX_COLOR_VALUE;
    YELLOW.B = 0;
    YELLOW.A = MAX_COLOR_VALUE;
}