//#pragma once
//
//#include "lib/memory/buffers.h"
//
//typedef struct {
//    int line_offset, offset,
//            offset_inc, inc,
//            offset_inc_secondary,
//            dx, run, run_twice,
//            dy, rise, rise_twice,
//            threshold, start, end,
//            error,
//            error_dec,
//            error_inc;
//    bool bottom_to_top;
//    bool left_to_right;
//} LineParams;
//static LineParams lp;
//
//#define DRAW_PIXEL(pixel) \
//if (lp.offset > 0 && \
//    lp.offset < frame_buffer.size) \
//    frame_buffer.pixels[lp.offset] = pixel
//
//// Optimized Bresenham's algorithm.
//void drawLine2D(int color, int x_start, int y_start, int x_end, int y_end) {
//    lp.offset = x_start + y_start * frame_buffer.width;
//
//    if (x_start == x_end && y_start == y_end) { // Draw single pixel:
//        DRAW_PIXEL(color);
//
//        return;
//    }
//
//    lp.bottom_to_top = y_start < y_end;
//    lp.left_to_right = x_start < x_end;
//
//    lp.dx = lp.left_to_right ? 1 : -1;
//    lp.dy = lp.bottom_to_top ? 1 : -1;
//    lp.line_offset = lp.bottom_to_top ? frame_buffer.width : -frame_buffer.width;
//
//    lp.run  = lp.left_to_right ? x_end - x_start : x_start - x_end;
//    lp.rise = lp.bottom_to_top ? y_end - y_start : y_start - y_end;
//
//    lp.offset_inc = lp.dx;
//
//    if (lp.rise == lp.run || !lp.rise) { // Draw a diagonal/horizontal line:
//        if (lp.rise) // Draw a diagonal line:
//            lp.offset_inc += lp.line_offset;
//        for (int i = x_start; i != x_end; i += lp.dx, lp.offset += lp.offset_inc)
//            DRAW_PIXEL(color);
//
//        return;
//    } else if (!lp.run) { // Draw a vertical line:
//        for (int i = y_start; i != y_end; i += lp.dy, lp.offset += lp.line_offset)
//            DRAW_PIXEL(color);
//
//        return;
//    }
//
//    // Draw any non-trivial line (by default, configure for a shallow line):
//    lp.start = x_start;
//    lp.end = x_end;
//    lp.inc = lp.dx;
//    lp.rise_twice = lp.rise + lp.rise;
//    lp.run_twice  = lp.run + lp.run;
//    lp.threshold = lp.run;
//    lp.error_dec = lp.run_twice;
//    lp.error_inc = lp.rise_twice;
//    lp.offset_inc_secondary = lp.line_offset;
//
//    if (lp.rise > lp.run) { // Reconfigure for a steep line:
//        lp.start = y_start;
//        lp.end = y_end;
//        lp.offset_inc_secondary = lp.dx;
//        lp.inc = lp.dy;
//        lp.offset_inc = lp.line_offset;
//        lp.threshold = lp.rise;
//        lp.error_dec = lp.rise_twice;
//        lp.error_inc = lp.run_twice;
//    }
//
//    lp.error = 0;
//    for (int i = lp.start; i != lp.end; i += lp.inc) {
//        DRAW_PIXEL(color);
//        lp.offset += lp.offset_inc;
//        lp.error += lp.error_inc;
//        if (lp.error > lp.threshold) {
//            lp.error -= lp.error_dec;
//            lp.offset += lp.offset_inc_secondary;
//        }
//    }
//}