#include "renderer.h"

void render(int width, int height, u32* pixels) {
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            *pixels++ = ((y << 16) | x);
}