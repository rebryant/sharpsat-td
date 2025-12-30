/*========================================================================
  Copyright (c) 2025 Randal E. Bryant, Carnegie Mellon University
  
  Permission is hereby granted, free of
  charge, to any person obtaining a copy of this software and
  associated documentation files (the "Software"), to deal in the
  Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom
  the Software is furnished to do so, subject to the following
  conditions:
  
  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ========================================================================*/

#pragma once

#include <stdint.h>

#ifdef __CUDACC__
#ifndef CUDA_ALL
#define CUDA_ALL __host__ __device__
#define CUDA_HOST __host__
#endif
#else
#define CUDA_ALL
#define CUDA_HOST
#endif

/********************* Double Properties **********************/

typedef double fp64_t;
typedef float fp32_t;


// Format parameters
#define FP64_EXP_OFFSET 52
#define FP64_SIGN_OFFSET 63
#define FP64_EXP_MASK ((uint64_t) 0x7ff)
#define FP64_BIAS ((int64_t) 0x3ff)
#define FP64_MAX_EXPONENT (FP64_EXP_MASK - FP64_BIAS - 1)

#define FMA64(x, y, z) fma((x), (y), (z))

// Minimum difference in exponents to ignore one argument to addition
#define FP64_MAX_PREC 55

// Format parameters
#define FP32_EXP_OFFSET 23
#define FP32_SIGN_OFFSET 31
#define FP32_EXP_MASK ((uint32_t) 0xff)
#define FP32_BIAS ((int32_t) 0x7f)
#define FP32_MAX_EXPONENT (FP32_EXP_MASK - FP32_BIAS - 1)

// Minimum difference in exponents to ignore one argument to addition
#define FP32_MAX_PREC 26

#define FMA32(x, y, z) fmaf((x), (y), (z))

/******************* FP64 *************************/

CUDA_ALL
static uint64_t fp64_get_bits(fp64_t x) {
    union {
	fp64_t   d;
	uint64_t b;
    } u;
    u.d = x;
    return u.b;
}


CUDA_ALL
static fp64_t fp64_from_bits(uint64_t bx) {
    union {
	fp64_t   d;
	uint64_t b;
    } u;
    u.b = bx;
    return u.d;
}

/* Get exponent as unsigned integer */
CUDA_ALL
static uint64_t fp64_get_biased_exponent(fp64_t x) {
    uint64_t bx = fp64_get_bits(x);
    return (bx >> FP64_EXP_OFFSET) & FP64_EXP_MASK;
}

/* Get exponent as signed integer */
CUDA_ALL
static int64_t fp64_get_exponent(fp64_t x) {
    int64_t bexp = (int64_t) fp64_get_biased_exponent(x);
    return bexp - FP64_BIAS;
}

CUDA_ALL
static uint64_t fp64_get_sign(fp64_t x) {
    uint64_t bx = fp64_get_bits(x);
    return (bx >> FP64_SIGN_OFFSET) & 0x1;
}

CUDA_ALL
static uint64_t fp64_get_fraction(fp64_t x) {
    uint64_t bx = fp64_get_bits(x);
    uint64_t umask = ((int64_t) 1 << FP64_EXP_OFFSET) - 1;
    return bx & umask;
}

/* Signed exponent too small */
CUDA_ALL
static bool fp64_exponent_below(int64_t exp) {
    return exp <= -(int64_t) FP64_BIAS;
}

/* Signed exponent too large */
CUDA_ALL
static bool fp64_exponent_above(int64_t exp) {
    return exp >= (int64_t) FP64_EXP_MASK - FP64_BIAS;
}

CUDA_ALL
static fp64_t fp64_assemble(uint64_t sign, int64_t exp, uint64_t frac) {
    int64_t bexp = exp + FP64_BIAS;
    uint64_t bx = frac;
    bx += bexp << FP64_EXP_OFFSET;
    bx += sign << FP64_SIGN_OFFSET;
    return fp64_from_bits(bx);
}

CUDA_ALL
static fp64_t fp64_replace_exponent(fp64_t x, int64_t exp) {
    uint64_t bexp = (uint64_t) (exp + FP64_BIAS);
    bexp = bexp << FP64_EXP_OFFSET;
    uint64_t bx = fp64_get_bits(x);
    uint64_t mask = ~(FP64_EXP_MASK << FP64_EXP_OFFSET);
    bx &= mask;
    bx += bexp;
    return fp64_from_bits(bx);
}

/* 
   Shift exponent up or down.
   Assume only risk is underflow
*/
CUDA_ALL
static fp64_t fp64_shift_exponent(fp64_t x, int64_t shift) {
    int64_t nexp = fp64_get_exponent(x) + shift;
    if (fp64_exponent_below(nexp))
	return 0.0;
    return fp64_replace_exponent(x, nexp);
}

CUDA_ALL
static fp64_t fp64_zero_exponent(fp64_t x) {
    uint64_t bexp = (uint64_t) FP64_BIAS << FP64_EXP_OFFSET;
    uint64_t bx = fp64_get_bits(x);
    uint64_t mask = ~(FP64_EXP_MASK << FP64_EXP_OFFSET);
    bx &= mask;
    bx += bexp;
    return fp64_from_bits(bx);
}

CUDA_ALL
static fp64_t fp64_infinity(int sign) {
    return fp64_assemble(sign, FP64_EXP_MASK - FP64_BIAS, 0);
}

/* 
   Represent power of 2.
   Watch for underflow (without using subnormal)
   but don't worry about overflow
*/

CUDA_ALL
static fp64_t fp64_power2(int64_t p) {
    if (p <= -(int64_t) FP64_BIAS)
	return 0.0;
    else
	return fp64_assemble(0, p, 0);
}


/******************* FP32 *************************/

CUDA_ALL
static uint32_t fp32_get_bits(fp32_t x) {
    union {
	fp32_t   d;
	uint32_t b;
    } u;
    u.d = x;
    return u.b;
}


CUDA_ALL
static fp32_t fp32_from_bits(uint32_t bx) {
    union {
	fp32_t   d;
	uint32_t b;
    } u;
    u.b = bx;
    return u.d;
}

/* Get exponent as unsigned integer */
CUDA_ALL
static uint32_t fp32_get_biased_exponent(fp32_t x) {
    uint32_t bx = fp32_get_bits(x);
    return (bx >> FP32_EXP_OFFSET) & FP32_EXP_MASK;
}

/* Get exponent as signed integer */
CUDA_ALL
static int32_t fp32_get_exponent(fp32_t x) {
    int32_t bexp = (int32_t) fp32_get_biased_exponent(x);
    return bexp - FP32_BIAS;
}

CUDA_ALL
static uint32_t fp32_get_sign(fp32_t x) {
    uint32_t bx = fp32_get_bits(x);
    return (bx >> FP32_SIGN_OFFSET) & 0x1;
}

CUDA_ALL
static uint32_t fp32_get_fraction(fp32_t x) {
    uint32_t bx = fp32_get_bits(x);
    uint32_t umask = ((int32_t) 1 << FP32_EXP_OFFSET) - 1;
    return bx & umask;
}

/* Signed exponent too small */
CUDA_ALL
static bool fp32_exponent_below(int32_t exp) {
    return exp <= -(int32_t) FP32_BIAS;
}

/* Signed exponent too large */
CUDA_ALL
static bool fp32_exponent_above(int32_t exp) {
    return exp >= (int32_t) FP32_EXP_MASK - FP32_BIAS;
}

CUDA_ALL
static fp32_t fp32_assemble(uint32_t sign, int32_t exp, uint32_t frac) {
    int32_t bexp = exp + FP32_BIAS;
    uint32_t bx = frac;
    bx += bexp << FP32_EXP_OFFSET;
    bx += sign << FP32_SIGN_OFFSET;
    return fp32_from_bits(bx);
}

CUDA_ALL
static fp32_t fp32_replace_exponent(fp32_t x, int32_t exp) {
    uint32_t bexp = (uint32_t) (exp + FP32_BIAS);
    bexp = bexp << FP32_EXP_OFFSET;
    uint32_t bx = fp32_get_bits(x);
    uint32_t mask = ~(FP32_EXP_MASK << FP32_EXP_OFFSET);
    bx &= mask;
    bx += bexp;
    return fp32_from_bits(bx);
}

/* 
   Shift exponent up or down.
   Assume only risk is underflow
*/
CUDA_ALL
static fp32_t fp32_shift_exponent(fp32_t x, int32_t shift) {
    int32_t nexp = fp32_get_exponent(x) + shift;
    if (fp32_exponent_below(nexp))
	return 0.0;
    return fp32_replace_exponent(x, nexp);
}


CUDA_ALL
static fp32_t fp32_zero_exponent(fp32_t x) {
    uint32_t bexp = (uint32_t) FP32_BIAS << FP32_EXP_OFFSET;
    uint32_t bx = fp32_get_bits(x);
    uint32_t mask = ~(FP32_EXP_MASK << FP32_EXP_OFFSET);
    bx &= mask;
    bx += bexp;
    return fp32_from_bits(bx);
}

CUDA_ALL
static fp32_t fp32_infinity(int sign) {
    return fp32_assemble(sign, FP32_EXP_MASK - FP32_BIAS, 0);
}


/* 
   Represent power of 2.
   Watch for underflow (without using subnormal)
   but don't worry about overflow
*/
CUDA_ALL
static fp32_t fp32_power2(int p) {
    if (p <= -(int32_t) FP32_BIAS)
	return 0.0;
    else
	return fp32_assemble(0, p, 0);
}
