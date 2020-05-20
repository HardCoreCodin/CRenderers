#pragma once

#ifdef PLATFORM_IS_WINDOWS



#endif

#include "lib/core/types.h"
#include "lib/core/string.h"
#include "lib/memory/allocators.h"

Perf* createPerf(GetTicks getTicks, u64 ticks_per_second) {
    Perf* perf = Alloc(Perf);

    perf->getTicks = getTicks;
    perf->delta.ticks = 0;
    perf->delta.seconds = 0;

    perf->accum.ticks = 0;
    perf->accum.frames = 0;

    perf->avg.ticks_per_frame = 0;
    perf->avg.frames_per_tick = 0;

    perf->ticks.after = 0;
    perf->ticks.before = getTicks();
    perf->ticks_per_second = ticks_per_second;
    perf->seconds_per_tick = 1.0f / perf->ticks_per_second;
    perf->milliseconds_per_tick = 1000.0f / perf->ticks_per_second;
    perf->microseconds_per_tick = 1000 * perf->milliseconds_per_tick;
    perf->nanoseconds_per_tick = 1000 * perf->microseconds_per_tick;
    perf->ticks_per_interval = perf->ticks_per_second / 4;

    return perf;
}

inline void perfAccum(Perf* perf) {
    perf->accum.ticks += perf->delta.ticks;
    perf->accum.frames++;
}

inline void perfSum(Perf* perf) {
    perf->avg.frames_per_tick = (f64)perf->accum.frames / perf->accum.ticks;
    perf->avg.ticks_per_frame = (f64)perf->accum.ticks / perf->accum.frames;
    perf->avg.frames_per_second = (u16)(perf->avg.frames_per_tick * perf->ticks_per_second);
    perf->avg.milliseconds_per_frame = (u16)(perf->avg.ticks_per_frame * perf->milliseconds_per_tick);
    perf->avg.microseconds_per_frame = (u16)(perf->avg.ticks_per_frame * perf->microseconds_per_tick);
    perf->avg.nanoseconds_per_frame = (u16)(perf->avg.ticks_per_frame * perf->nanoseconds_per_tick);
    perf->accum.ticks = perf->accum.frames = 0;
}

inline void perfStart(Perf* perf) {
    perf->ticks.before = perf->getTicks();
}

inline void perfEnd(Perf* perf) {
    perf->ticks.after = perf->getTicks();
    perf->delta.ticks = perf->ticks.after - perf->ticks.before;
    perfAccum(perf);
    perfSum(perf);
}

inline void perfStartFrame(Perf* perf) {
    perf->ticks.after = perf->ticks.before;
    perf->ticks.before = perf->getTicks();
    perf->delta.ticks = perf->ticks.before - perf->ticks.after;
    perf->delta.seconds = perf->delta.ticks * perf->seconds_per_tick;
}

inline void perfEndFrame(Perf* perf) {
    perf->ticks.after = perf->getTicks();
    perf->delta.ticks = perf->ticks.after - perf->ticks.before;
    perfAccum(perf);
    if (perf->accum.ticks >= perf->ticks_per_interval) perfSum(perf);
}

inline void perfOut(Perf* perf, HUD* hud) {
    if (!perf->accum.ticks) printNumberIntoString(perf->avg.nanoseconds_per_frame, hud->perf);
}