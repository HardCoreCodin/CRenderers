#pragma once

#include "lib/core/types.h"

#ifdef __CUDACC__
__device__
__host__
__forceinline__
#else
inline
#endif
void approach(f32* current_velocity, f32 target_velocity, f32 change_in_velocity) {
    if (target_velocity > *current_velocity + change_in_velocity)
        *current_velocity += change_in_velocity;

    else if (target_velocity < *current_velocity - change_in_velocity)
        *current_velocity -= change_in_velocity;

    else
        *current_velocity = target_velocity;
}