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

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>

#include "fp.h"

/* 
   Optionally avoid any use of GMP libraries.
   Set EFP64_NO_GMP to 0 before including this file
*/
#ifndef EFP64_NO_GMP
#define EFP64_NO_GMP 0
#endif

#if !EFP64_NO_GMP
#include <gmp.h>
#endif


/*
  Representation of floating-point numbers based on fp64,
  but with additional exponent field to support extended range

  This version has lower bits of exponent in fp and upper bits in exp.
  based on ACM Algorithm 567.

  Key difference: Keep exp scaled.

*/
typedef struct {
    fp64_t fp; 
    int64_t exp; 
} efp64_t;

/********************* Defines **********************/ 

/* Divide x into high part and low part according to number of bits */
CUDA_ALL
static int64_t hi_part(int64_t x, int bits) {
    int64_t mask = ~(int64_t) 0 << bits;
    return x & mask;
}

CUDA_ALL
static int64_t lo_part(int64_t x, int bits) {
    int64_t mask = ((int64_t) 1 << bits) - 1;
    return x & mask;
}


/* Number of bits of exponent kept in fp */
#define EXP_BITS 6
#define EXP_MODULUS (1<<(EXP_BITS))

/* Exponent for zero.  Smaller than any other */
#define ZEXP (hi_part(INT64_MIN/2, EXP_BITS))


/* Max number of times fractions can be multiplied without overflowing exponent */
#define EFP64_MAX_MUL (FP64_MAX_EXPONENT/EXP_MODULUS-2)

/* Required size of buffer from printing EFP64 */
#define EFP64_BUF 100
/* Number of significant digits when printing EFP64 */
#define EFP64_NSIG 18


/********************* EFP64 *************************/

static void efp64_show(efp64_t val, FILE *out);

CUDA_ALL
static efp64_t efp64_collect(fp64_t fp, int64_t exp) {
    efp64_t nval;
    nval.fp = fp;
    nval.exp = exp;
    return nval;
}

CUDA_ALL
static void efp64_disperse(efp64_t a, fp64_t *dptr, int64_t *eptr) {
    *dptr = a.fp;
    *eptr = a.exp;
}

CUDA_ALL
static bool efp64_is_zero(efp64_t a) {
    return a.fp == 0.0;
}

CUDA_ALL
static efp64_t efp64_zero() {
    efp64_t nval;
    nval.fp = 0.0;
    nval.exp = ZEXP;
    return nval;
}

CUDA_ALL
static bool efp64_is_valid(efp64_t a) {
    return !isnan(a.fp) && !isinf(a.fp);
}

CUDA_ALL
static int64_t efp64_full_exponent(efp64_t val) {
    return val.exp + fp64_get_exponent(val.fp);
}
    
CUDA_ALL
static fp64_t efp64_zeroed_fraction(efp64_t val) {
    return fp64_zero_exponent(val.fp);
}

CUDA_ALL
static efp64_t efp64_canonicalize(efp64_t a) {
    if (efp64_is_zero(a))
	return efp64_zero();
    int64_t nexp = efp64_full_exponent(a);
    efp64_t nval;
    nval.exp = hi_part(nexp, EXP_BITS);
    nval.fp = fp64_replace_exponent(a.fp, lo_part(nexp, EXP_BITS));
    return nval;
}

CUDA_ALL
/* Canonicalize when exponent with +- 1 of target */
static efp64_t efp64_quick_canonicalize(efp64_t a) {
    efp64_t nval = a;
    fp64_t low = fp64_power2(-EXP_MODULUS);
    fp64_t high = fp64_power2(EXP_MODULUS);
    if (fabs(nval.fp) < 1.0) {
	nval.fp *= high;
	nval.exp -= EXP_MODULUS;
    } else if (fabs(nval.fp) >= high) {
	nval.fp *= low;
	nval.exp += EXP_MODULUS;
    }
    return nval;
}

CUDA_ALL
/* Canonicalize when exponent with + 1 of target */
static efp64_t efp64_quick_down_canonicalize(efp64_t a) {
    efp64_t nval = a;
    fp64_t low = fp64_power2(-EXP_MODULUS);
    fp64_t high = fp64_power2(EXP_MODULUS);
    if (fabs(nval.fp) >= high) {
	nval.fp *= low;
	nval.exp += EXP_MODULUS;
    }
    return nval;
}

CUDA_ALL
/* Canonicalize when exponent with - 1 of target */
static efp64_t efp64_quick_up_canonicalize(efp64_t a) {
    efp64_t nval = a;
    fp64_t high = fp64_power2(EXP_MODULUS);
    if (fabs(nval.fp) < 1.0) {
	nval.fp *= high;
	nval.exp -= EXP_MODULUS;
    }
    return nval;
}


CUDA_ALL
static efp64_t efp64_from_fp64(fp64_t dval) {
    efp64_t nval;
    nval.exp = 0;
    nval.fp = dval;
    return efp64_canonicalize(nval);
}

#if !EFP64_NO_GMP
CUDA_HOST
static efp64_t efp64_from_mpf(mpf_srcptr fval) {
    efp64_t nval;
    long int exp;
    nval.fp = mpf_get_d_2exp(&exp, fval);
    if (nval.fp == 0)
	return efp64_zero();
    nval.exp = (int64_t) exp;
    nval = efp64_canonicalize(nval);
    return nval;
}

CUDA_HOST
static void efp64_to_mpf(mpf_ptr dest, efp64_t eval) {
    mpf_set_d(dest, eval.fp);
    if (efp64_is_zero(eval))
	return;
    if (eval.exp < 0)
	mpf_div_2exp(dest, dest, -eval.exp);
    else if (eval.exp > 0)
	mpf_mul_2exp(dest, dest, eval.exp);
}
#endif /* !EFP64_NO_GMP */

CUDA_ALL
static fp64_t efp64_to_fp64(efp64_t eval) {
    if (efp64_is_zero(eval))
	return 0.0;
    int64_t full_exp = efp64_full_exponent(eval);
    if (fp64_exponent_below(full_exp))
	return 0.0;
    if (fp64_exponent_above(full_exp)) {
	int sign = fp64_get_sign(eval.fp);
	return fp64_infinity(sign);
    }
    return fp64_replace_exponent(eval.fp, full_exp);
}

CUDA_ALL
static unsigned efp64_to_unsigned(efp64_t a) {
    if (a.fp <= 0)
	return 0U;
    fp64_t d = efp64_to_fp64(a);
    if (d > UINT_MAX)
	return UINT_MAX;
    return (unsigned) d;
}

CUDA_ALL
static int efp64_to_int(efp64_t a) {
    fp64_t d = efp64_to_fp64(a);
    if (d >= INT_MAX)
	return INT_MAX;
    if (d <= INT_MIN)
	return INT_MIN;
    return (unsigned) d;
}

CUDA_ALL
static bool efp64_is_equal(efp64_t a, efp64_t b) {
    if (efp64_is_zero(a))
	return efp64_is_zero(b);
    return a.fp == b.fp && a.exp == b.exp;
}

CUDA_ALL
static efp64_t efp64_negate(efp64_t a) {
    efp64_t nval;
    if (efp64_is_zero(a))
	return a;
    nval.exp = a.exp;
    nval.fp = -a.fp;
    return nval;
}

CUDA_ALL
static efp64_t efp64_add(efp64_t a, efp64_t b) {
    efp64_t nval;
    if (a.exp > b.exp) {
	nval.exp = a.exp;
	nval.fp = FMA64(b.fp, fp64_power2(b.exp-a.exp), a.fp);
    } else {
	nval.exp = b.exp;
	nval.fp = FMA64(a.fp, fp64_power2(a.exp-b.exp), b.fp);
    }
    nval = efp64_quick_canonicalize(nval);
    return nval;
}

CUDA_ALL
static efp64_t efp64_quick_mul(efp64_t a, efp64_t b) {
    efp64_t nval;
    nval.exp = a.exp + b.exp;
    nval.fp = a.fp * b.fp;
    return nval;
}


CUDA_ALL
static efp64_t efp64_mul(efp64_t a, efp64_t b) {
    return efp64_quick_down_canonicalize(efp64_quick_mul(a, b));
}

/* Return correctly rounded version of a*b + c */
CUDA_ALL
static efp64_t efp64_fma(efp64_t a, efp64_t b, efp64_t c) {
    /* This isn't right */
    return efp64_add(efp64_mul(a, b), c);
}

CUDA_ALL
static efp64_t efp64_mul_seq_slow(efp64_t *val, int len) {
    efp64_t result = efp64_from_fp64(1.0);
    int i;
    for (i = 0; i < len; i++)
	result = efp64_mul(result, val[i]);
    return result;
}

CUDA_ALL
static efp64_t efp64_mul_seq_x1(efp64_t *val, int len) {
    if (len == 0)
	return efp64_from_fp64(1.0);
    efp64_t result = val[0];
    int i;
    int count = 1;
    for (i = 1; i < len; i++) {
	efp64_t arg = val[i];
	result = efp64_quick_mul(result, arg);
	if (++count > EFP64_MAX_MUL) {
	    count = 0;
	    result = efp64_canonicalize(result);
	}
    }
    return efp64_canonicalize(result);
}

CUDA_ALL
static efp64_t efp64_mul_seq_x4(efp64_t *val, int len) {
    // Assume len >= 4
    efp64_t prod[4];
    int i, j;
    for (j = 0; j < 4; j++) 
	prod[j] = val[j];
    int count = 0;
    for (i = 4; i <= len-4; i+= 4) {
	for (j = 0; j < 4; j++)
	    prod[j] = efp64_quick_mul(prod[j], val[i+j]);
	if (++count > EFP64_MAX_MUL) {
	    count = 0;
	    for (j = 0; j < 4; j++)
		prod[j] = efp64_canonicalize(prod[j]);
	}
    }
    if (count * 4 > EFP64_MAX_MUL) {
	for (j = 0; j < 4; j++)
	    prod[j] = efp64_canonicalize(prod[j]);
    }

    efp64_t result = prod[0];
    for (j = 1; j < 4; j++)
	result = efp64_quick_mul(result, prod[j]);
    for (; i < len; i++)
	result = efp64_quick_mul(result, val[i]);
    return efp64_canonicalize(result);
}

/* Compute product of sequence of values */
CUDA_ALL
static efp64_t efp64_mul_seq(efp64_t *val, int len) {
    if (len < 8)
	return efp64_mul_seq_x1(val, len);
    return efp64_mul_seq_x4(val, len);
}

CUDA_ALL
static efp64_t efp64_div(efp64_t a, efp64_t b) {
    efp64_t nval;
    nval.fp = a.fp / b.fp;
    nval.exp = a.exp - b.exp;
    return efp64_quick_up_canonicalize(nval);
}

CUDA_ALL
static int efp64_cmp(efp64_t a, efp64_t b) {
    int sa = a.fp < 0;
    int sb = b.fp < 0;
    if (sa && !sb)
	return -1;
    if (!sa && sb)
	return 1;
    int flip = sa ? -1 : 1;
    if (a.exp > b.exp)
	return flip;
    if (a.exp < b.exp)
	return -flip;
    if (a.fp < b.fp)
	return -1;
    if (a.fp > b.fp)
	return 1;
    return 0;
}


CUDA_ALL
static efp64_t efp64_sqrt(efp64_t a) {
    if (efp64_is_zero(a) | (a.fp < 0))
	return efp64_zero();
    efp64_t nval;
    nval.fp = sqrt(a.fp);
    nval.exp = a.exp / 2;
    return efp64_canonicalize(nval);
}

CUDA_ALL
static efp64_t efp64_scale_power2(efp64_t val, int64_t power) {
    efp64_t nval;
    nval.fp = val.fp;
    nval.exp = val.exp + power;
    return nval;
}

/**** I/O Support.  Only needed when not using GMP *******/

/* Create right-justified string representation of nonnegative number */
CUDA_ALL
static void rj_string(char *sbuf, long long val, int len) {
    int i;
    for (i = 0; i < len; i++)
	sbuf[i] = '0';
    sbuf[len] = 0;
    if (val <= 0)
	return;
    i = len-1;
    while(val) {
	sbuf[i--] = '0' + (val % 10);
	val = val / 10;
    }
}

/* Generate integral power of 10 */
CUDA_ALL
static long long p10(int exp) {
    if (exp < 0)
	return 0;
    long long result = 1;
    long long power = 10;
    while (exp != 0) {
	if (exp & 0x1)
	    result *= power;
	power *= power;
	exp = exp >> 1;
    }
    return result;
}

#if EFP64_NO_GMP
/* Buf must point to buffer with at least EFP64_BUF character capacity */
CUDA_HOST
static void efp64_string(efp64_t a, char *buf, int nsig) {
    char sbuf[25];
    if (nsig <= 0)
	nsig = 1;
    if (nsig > 20)
	nsig = 20;
    if (efp64_is_zero(a)) {
	snprintf(buf, EFP64_BUF, "0.0");
	return;
    }
    const char *sgn = "";
    fp64_t da = efp64_zeroed_fraction(a);
    int64_t de = efp64_full_exponent(a);
    if (da < 0) {
	da = -da;
	sgn = "-";
    }
    // Convert exponent to base 10
    fp64_t dlog = ((fp64_t) de) * log10(2.0);
    // Get integer part of exponent
    long long dec = (long long) floor(dlog);
    // Incorporate the fractional part of the exponent into da
    da *= __exp10(dlog-floor(dlog));
    // Get decimal exponent for da
    long long dexp = (long long) floor(log10(da));
    // Add to decimal exponent
    dec += dexp;
    // Scale da to become integer representation of final fraction
    da *= p10(nsig-1-dexp);
    // Round it
    long long dfrac = llround(da);
    // Get digits to the left and right of the decimal point
    long long sep = p10(nsig-1);
    long long lfrac = dfrac / sep;
    long long rfrac = dfrac % sep;
    rj_string(sbuf, rfrac, nsig-1);
    if (dec == 0) 
	snprintf(buf, EFP64_BUF, "%s%lld.%s", sgn, lfrac, sbuf);
    else if (dec > 0)
	snprintf(buf, EFP64_BUF, "%s%lld.%se+%lld", sgn, lfrac, sbuf, dec);
    else
	snprintf(buf, EFP64_BUF, "%s%lld.%se%lld", sgn, lfrac, sbuf, dec);
}
#endif

/* 
   Logarithms.
   Relies on library log2 function,
   which isn't very accurate when a is close to 1.0 
*/
CUDA_ALL
static fp64_t efp64_log2d(efp64_t a) {
    fp64_t d = efp64_zeroed_fraction(a);
    if (d <= 0)
	return 0.0;
    int64_t e = efp64_full_exponent(a);
    if (d == 1.0)
	return (fp64_t) e;
    if (!fp64_exponent_below(e) && !fp64_exponent_above(e))
	return log2(efp64_to_fp64(a));
    fp64_t log_weight, dlog;
    if (e < 0) {
	log_weight = -1.0;
	dlog = -log2(d/2); // Force to have negative log
	e = -(e+1);
    } else {
	log_weight = 1.0;
	dlog = log2(d);
    }
    // Track case where original log computation underflowed to zero
    int uflow = dlog == 0;
    // Construct unsigned 64-bit value representing logarithm
    // normalized to have MSB=1
    uint64_t log_val = e;

    while ((log_val >> 63) == 0) {
	log_val *= 2;
	dlog *= 2;
	if (dlog >= 1) {
	    log_val++;
	    dlog = dlog-1;
	}
	log_weight *= 0.5;
    }
    if (uflow || dlog != 0)
	// Set LSB to 1 break RN tie
	log_val = log_val | 0x1;

    return ((fp64_t) log_val) * log_weight;
}

CUDA_ALL
static efp64_t efp64_log2(efp64_t a) {
    return efp64_from_fp64(efp64_log2d(a));
}

CUDA_ALL
static efp64_t efp64_log10(efp64_t a) {
    fp64_t d10 = log10(2.0);
    return efp64_mul(efp64_log2(a),
		   efp64_from_fp64(d10));
}

/*
  EFP64 implementations of functions.  Not very accurate
*/


/* Generate efp64 power of x */
CUDA_ALL
static efp64_t xpe(fp64_t x, int64_t exp) {
    if (x == 0)
	return efp64_zero();
    if (exp < 0) {
	exp = -exp;
	x = 1.0/x;
    }
    efp64_t power = efp64_from_fp64(x);
    efp64_t result = efp64_from_fp64(1.0);
    while (exp != 0) {
	if (exp & 0x1)
	    result = efp64_mul(result, power);
	power = efp64_mul(power, power);
	exp = exp >> 1;
    }
    return result;
}


/* Generate efp64 power of 10 */
CUDA_ALL
static efp64_t ep10(int64_t exp) {
    return xpe(10.0, exp);
}


CUDA_HOST
static int efp64_sscanf(const char *s, efp64_t *result) {
    char buf[50];
    efp64_t nval = efp64_zero();
    int i = 0;
    while (i < sizeof(buf) && *s != 0 && *s != 'e' && *s != 'E')
	buf[i++] = *s++;
    buf[i++] = 0;
    bool ok = sscanf(buf, "%lf", &nval.fp) > 0;
    if (ok && *s != 0) {
	long long lexp = 0;
	ok = sscanf(s+1, "%lld", &lexp) > 0;
	efp64_t p10 = ep10(lexp);
	nval = efp64_mul(nval, p10);
    }
    if (ok) 
	*result = efp64_canonicalize(nval);
    else
	*result = efp64_zero();
    return (int) ok;
}

CUDA_HOST
static int efp64_fscanf(FILE *infile, efp64_t *result) {
    char buf[100];
    int c;
    while ((c = fgetc(infile)) != EOF && isspace(c))
	;
    int i = 0;
    while ((c = fgetc(infile)) != EOF && i < sizeof(buf) &&
	   !isspace(c) && (isdigit(c) || c == 'e' || c == 'E' || c == '.'))
	buf[i++] = c;
    buf[i] = 0;
    return efp64_sscanf((const char *) buf, result);
}

CUDA_HOST
static void efp64_show(efp64_t val, FILE *out) {
    fprintf(out, "[2^%ld * %g]", (long) val.exp, val.fp);
}


