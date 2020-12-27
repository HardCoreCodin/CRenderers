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
      BLACK;

void initAppGlobals() {
    WHITE.R = 255;
    WHITE.G = 255;
    WHITE.B = 255;
    WHITE.A = 0;

    BLACK.R = 0;
    BLACK.G = 0;
    BLACK.B = 0;
    BLACK.A = 0;
}