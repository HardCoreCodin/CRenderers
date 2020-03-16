#pragma once

#ifndef __cplusplus
#define false 0
#define true 1
typedef unsigned char      bool;
#endif

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

typedef float  f32;
typedef double f64;

f32 t2, c, s, f;

void set_point_on_unit_circle(f32 t) {
	t2 = t * t;
	f = 1 / (1 + t2);
	c = f - f * t2;
	s = f * 2 * t; 
}