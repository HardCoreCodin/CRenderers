#pragma once

#define PERF 1

#ifdef PLATFORM_IS_WINDOWS

static LARGE_INTEGER perf_counter;
static LARGE_INTEGER performance_frequency;

#define GET_TICKS_PER_SECOND(t) QueryPerformanceFrequency(&performance_frequency); t = performance_frequency.QuadPart
#define GET_TICKS(t) QueryPerformanceCounter(&perf_counter); t = (u64)perf_counter.QuadPart
#define OUT_STRING(s) OutputDebugStringA(s)

#endif

#include "lib/core/types.h"
#include "lib/core/string.h"

typedef struct PerfTicks {u64 current, last;} PerfTicks;
typedef struct PerfDelta {u64 ticks; f64 seconds;} PerfDelta;
typedef struct PerfAccum {u64 ticks, frames;} PerfAccum;
typedef struct PerfAvg {
    f64 frames_per_tick, ticks_per_frame;
    u16 frames_per_second, milliseconds_per_frame;
} PerfAvg;

typedef struct Perf {
    PerfDelta delta;
    PerfAccum accum;
    PerfTicks ticks;
    PerfAvg avg;
    u64 ticks_per_interval, ticks_per_second;
    f64 seconds_per_tick, milliseconds_per_tick;
} Perf;

void initPerf(Perf* p) {
    p->delta.ticks = 0;
    p->delta.seconds = 0;

    p->accum.ticks = 0;
    p->accum.frames = 0;

    p->avg.ticks_per_frame = 0;
    p->avg.frames_per_tick = 0;

    p->ticks.last = 0;
    GET_TICKS(p->ticks.current);
    GET_TICKS_PER_SECOND(p->ticks_per_second);
    p->seconds_per_tick = 1.0f / p->ticks_per_second;
    p->milliseconds_per_tick = 1000.0f / p->ticks_per_second;
    p->ticks_per_interval = p->ticks_per_second / 4;
}

#define PERF_ACCUM(p) \
    p.accum.ticks += p.delta.ticks; \
    p.accum.frames++;

#define PERF_SUM(p) \
    p.avg.frames_per_tick = (f64)p.accum.frames / p.accum.ticks; \
    p.avg.ticks_per_frame = (f64)p.accum.ticks / p.accum.frames; \
    p.avg.frames_per_second = (u16)(p.avg.frames_per_tick * p.ticks_per_second); \
    p.avg.milliseconds_per_frame = (u16)(p.avg.ticks_per_frame * p.milliseconds_per_tick); \
    p.accum.ticks = p.accum.frames = 0;

#define PERF_START_FRAME(p) \
    p.ticks.last = p.ticks.current; \
    GET_TICKS(p.ticks.current); \
    p.delta.ticks = p.ticks.current - p.ticks.last; \
    p.delta.seconds = p.delta.ticks * p.seconds_per_tick; \
    PERF_ACCUM(p);

#define PERF_FRAME_END(p) \
    if (p.accum.ticks >= p.ticks_per_interval) { \
        PERF_SUM(p) \
    }

#define PERF_START(p) GET_TICKS(p.ticks.last);
#define PERF_END(p) GET_TICKS(p.ticks.current); \
    p.delta.ticks = p.ticks.current - p.ticks.last; \
    PERF_ACCUM(p); \
    PERF_SUM(p);


#define PERF_OUT(p) if (!p.accum.ticks) printNumberIntoString(p.avg.milliseconds_per_frame, hud.perf);