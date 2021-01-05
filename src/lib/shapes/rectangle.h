#pragma once

#include "lib/core/types.h"
#include "lib/globals/display.h"
#include "line.h"

inline void drawRect(Bounds2Di *rect, Pixel pixel) {
    drawHLine2D(rect->x_range.min, rect->x_range.max, rect->y_range.min, pixel);
    drawHLine2D(rect->x_range.min, rect->x_range.max, rect->y_range.max, pixel);
    drawVLine2D(rect->y_range.min, rect->y_range.max, rect->x_range.min, pixel);
    drawVLine2D(rect->y_range.min, rect->y_range.max, rect->x_range.max, pixel);
}

inline void fillRect(Bounds2Di *rect, Pixel pixel) {
    if (abs(rect->x_range.max - rect->x_range.min) > abs(rect->y_range.max - rect->y_range.min))
        for (u16 y = rect->y_range.min; y <= rect->y_range.max; y++) drawHLine2D(rect->x_range.min, rect->x_range.max, y, pixel);
    else
        for (u16 x = rect->x_range.min; x <= rect->x_range.max; x++) drawVLine2D(rect->y_range.min, rect->y_range.max, x, pixel);
}
