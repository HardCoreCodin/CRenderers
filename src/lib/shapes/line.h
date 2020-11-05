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

//typedef unsigned char byte;
//
//void draw_line(int x0, int y0, int x1, int y1, u32 color) {
//    int index = x0 + y0 * window_width;
//    if (x0 == x1 && y0 == y1) { // Draw single pixel:
//        DRAW_PIXEL(index, color);
//        return;
//    }
//
//    int dx = 1;
//    int dy = 1;
//    int run  = x1 - x0;
//    int rise = y1 - y0;
//    if (x0 > x1) {
//        dx = -1;
//        run  = x0 - x1;
//    }
//
//    int index_inc_per_line = window_width;
//    if (y0 > y1) {
//        dy = -1;
//        rise = y0 - y1;
//        index_inc_per_line = -window_width;
//    }
//
//    // Configure for a trivial line (horizontal, vertical or diagonal, default to a shallow line):
//    int inc = dx;
//    int start = x0;
//    int end = x1 + inc;
//    int index_inc = dx;
//    if (rise > run) { // Reconfigure for a steep line:
//        inc = dy;
//        start = y0;
//        end = y1 + inc;
//        index_inc = index_inc_per_line;
//    }
//
//    if (rise == run || !rise || !run) { // Draw a trivial line:
//        if (rise && run) // Reconfigure for a diagonal line:
//            index_inc = index_inc_per_line + dx;
//
//        for (int i = start; i != end; i += inc, index += index_inc)
//            DRAW_PIXEL(index, color);
//
//        return;
//    }
//
//    // Configure for a non-trivial line (default to a shallow line):
//    int rise_twice = rise + rise;
//    int run_twice  = run + run;
//    int threshold = run;
//    int error_dec = run_twice;
//    int error_inc = rise_twice;
//    int secondary_inc = index_inc_per_line;
//    if (rise > run) { // Reconfigure for a steep line:
//        secondary_inc = dx;
//        threshold = rise;
//        error_dec = rise_twice;
//        error_inc = run_twice;
//    }
//
//    int error = 0;
//    for (int i = start; i != end; i += inc) {
//        DRAW_PIXEL(index, color);
//        index += index_inc;
//        error += error_inc;
//        if (error > threshold) {
//            error -= error_dec;
//            index += secondary_inc;
//        }
//    }
//}
//
//
//void drawTriangle(float* X, float* Y, int pitch, u32 color, u32* pixels) {
//    float dx1, x1, y1, xs,
//            dx2, x2, y2, xe,
//            dx3, x3, y3, dy;
//    int offset,
//            x, x1i, y1i, x2i, xsi, ysi = 0,
//            y, y2i, x3i, y3i, xei, yei = 0;
//    for (int i = 1; i <= 2; i++) {
//        if (Y[i] < Y[ysi]) ysi = i;
//        if (Y[i] > Y[yei]) yei = i;
//    }
//    byte* id = ysi ? (ysi == 1 ?
//                      (byte[3]){1, 2, 0} :
//                      (byte[3]){2, 0, 1}) :
//               (byte[3]){0, 1, 2};
//    x1 = X[id[0]]; y1 = Y[id[0]]; x1i = (int)x1; y1i = (int)y1;
//    x2 = X[id[1]]; y2 = Y[id[1]]; x2i = (int)x2; y2i = (int)y2;
//    x3 = X[id[2]]; y3 = Y[id[2]]; x3i = (int)x3; y3i = (int)y3;
//    dx1 = x1i == x2i || y1i == y2i ? 0 : (x2 - x1) / (y2 - y1);
//    dx2 = x2i == x3i || y2i == y3i ? 0 : (x3 - x2) / (y3 - y2);
//    dx3 = x1i == x3i || y1i == y3i ? 0 : (x3 - x1) / (y3 - y1);
//    dy = 1 - (y1 - (float)y1);
//    xs = dx3 ? x1 + dx3 * dy : x1; ysi = (int)Y[ysi];
//    xe = dx1 ? x1 + dx1 * dy : x1; yei = (int)Y[yei];
//    offset = pitch * y1i;
//    for (y = ysi; y < yei; y++){
//        if (y == y3i) xs = dx2 ? (x3 + dx2 * (1 - (y3 - (float)y3i))) : x3;
//        if (y == y2i) xe = dx2 ? (x2 + dx2 * (1 - (y2 - (float)y2i))) : x2;
//        xsi = (int)xs;
//        xei = (int)xe;
//        for (x = xsi; x < xei; x++) pixels[offset + x] = color;
//        offset += pitch;
//        xs += y < y3i ? dx3 : dx2;
//        xe += y < y2i ? dx1 : dx2;
//    }
//}
//
//float triangles_x[3] = {120.7f, 220.3f, 320.4f};
//float triangles_y[3] = {200.5f, 158.2f, 200.6f};
