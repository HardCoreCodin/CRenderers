#pragma once
#include "lib/core/types.h"

inline f32 approach(f32 from, f32 to, f32 step) {
    f32 delta = to - from;
    if (delta > step) return from + step;
    if (delta < -step) return from - step;
    return to;
}