#pragma once

#include "lib/core/types.h"

typedef u64 (*GetTicks)();
GetTicks getTicks;

u64 ticks_per_second;
f64 seconds_per_tick,
    milliseconds_per_tick,
    microseconds_per_tick,
    nanoseconds_per_tick;

typedef struct {
    f32 delta_time;
    u64 ticks_before,
        ticks_after,
        ticks_diff,
        accumulated_ticks,
        accumulated_frame_count,
        ticks_of_last_report,
        seconds,
        milliseconds,
        microseconds,
        nanoseconds;
    f64 average_frames_per_tick,
        average_ticks_per_frame;
    u16 average_frames_per_second,
        average_milliseconds_per_frame,
        average_microseconds_per_frame,
        average_nanoseconds_per_frame;
} Timer;
Timer render_timer,
      update_timer,
      aux_timer;