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

#include "efp64-c.h"


#if !EFP64_NO_GMP
#include "gmpxx.h"
#include "mpf-util.h"
#endif


#include <iostream>
#include <vector>



class EFP64 {
private:
    efp64_t eval;

public:

CUDA_ALL
    EFP64(efp64_t val) { eval = val; }

CUDA_ALL
    efp64_t& get_efp64_t() { return eval; }

CUDA_ALL
    EFP64() { eval = efp64_zero(); }

CUDA_ALL
    EFP64(fp64_t d) { eval = efp64_from_fp64(d); }

CUDA_ALL
    EFP64(fp32_t d) { eval = efp64_from_fp64((fp64_t) d); }

CUDA_ALL
    EFP64(int i) { eval = efp64_from_fp64((fp64_t) i); }

#if !EFP64_NO_GMP
CUDA_HOST
    EFP64(mpf_srcptr mval) { eval = efp64_from_mpf(mval); }

CUDA_HOST
    EFP64(mpf_class mval) { eval = efp64_from_mpf(mval.get_mpf_t()); }

CUDA_HOST
    void get_mpf(mpf_ptr dest) const { efp64_to_mpf(dest, eval); }

CUDA_HOST
    mpf_class get_mpf() const { mpf_class val; efp64_to_mpf(val.get_mpf_t(), eval); return val; }
#endif

CUDA_ALL
    bool is_zero() const { return efp64_is_zero(eval); }

CUDA_ALL
    bool is_valid() const {return efp64_is_valid(eval); }

CUDA_ALL
    fp64_t get_fp64() const { return efp64_to_fp64(eval); }

CUDA_ALL
    EFP64 add(const EFP64 &other) const { return EFP64(efp64_add(eval, other.eval)); }

CUDA_ALL
    EFP64 mul(const EFP64 &other) const { return EFP64(efp64_mul(eval, other.eval)); }

CUDA_ALL
    EFP64 fma(const EFP64 &other1, const EFP64 &other2) const { return EFP64(efp64_fma(eval, other1.eval, other2.eval)); }

CUDA_ALL
    EFP64 log2() const { return efp64_log2(eval); }

CUDA_ALL
    EFP64 log10() const { return efp64_log10(eval); }

    //CUDA_ALL
    //    EFP64& operator=(const EFP64 &v) { eval = v.eval; return *this; }
#if !EFP64_NO_GMP
CUDA_HOST
    EFP64& operator=(const mpf_t v) { eval = efp64_from_mpf(v); return *this; }
CUDA_HOST
    EFP64& operator=(const mpf_class &v) { eval = efp64_from_mpf(v.get_mpf_t()); return *this; }
#endif
    EFP64& operator=(const fp64_t v) { eval = efp64_from_fp64(v); return *this; }
CUDA_ALL
    EFP64& operator=(const unsigned long int v) { eval = efp64_from_fp64((fp64_t) v); return *this; }
CUDA_ALL
    EFP64& operator=(const unsigned long long int v) { eval = efp64_from_fp64((fp64_t) v); return *this; }
CUDA_ALL
    EFP64& operator=(const long long int v)  { eval = efp64_from_fp64((fp64_t) v); return *this; }
CUDA_ALL
    EFP64& operator=(const unsigned int v)  { eval = efp64_from_fp64((fp64_t) v); return *this; }
CUDA_ALL
    EFP64& operator=(const long int v)  { eval = efp64_from_fp64((fp64_t) v); return *this; }
CUDA_ALL
    EFP64& operator=(const int v)  { eval = efp64_from_fp64((fp64_t) v); return *this; }

CUDA_ALL
    bool operator==(const EFP64 &other) const { return efp64_is_equal(eval, other.eval); }
CUDA_ALL
    bool operator!=(const EFP64 &other) const { return !efp64_is_equal(eval, other.eval); }
CUDA_ALL
    EFP64 operator+(const EFP64 &other) const { return EFP64(efp64_add(eval, other.eval)); }
CUDA_ALL
    EFP64 operator*(const EFP64 &other) const { return EFP64(efp64_mul(eval, other.eval)); }
CUDA_ALL
    EFP64 operator*(const fp64_t &other) const { return EFP64(efp64_mul(eval, efp64_from_fp64(other))); }
CUDA_ALL
    EFP64 operator/(const EFP64 &other) const { return EFP64(efp64_div(eval, other.eval)); }
CUDA_ALL
    EFP64 operator-() const { return EFP64(efp64_negate(eval)); }
CUDA_ALL
    EFP64 operator-(const EFP64 &other) const { return EFP64(efp64_add(eval, efp64_negate(other.eval))); }
CUDA_ALL
    EFP64 operator<<(int64_t power) const { return EFP64(efp64_scale_power2(eval, power)); }
CUDA_ALL
    EFP64 operator>>(int64_t power) const { return EFP64(efp64_scale_power2(eval, -power)); }
CUDA_ALL
    EFP64& operator*=(const EFP64 &other) { eval = efp64_mul(eval, other.eval); return *this; }
CUDA_ALL
    EFP64& operator*=(const fp64_t &other) { eval = efp64_mul(eval, efp64_from_fp64(other)); return *this; }
CUDA_ALL
    EFP64& operator+=(const EFP64 &other) { eval = efp64_add(eval, other.eval); return *this; }
CUDA_ALL
    EFP64& operator/=(const EFP64 &other) { eval = efp64_div(eval, other.eval); return *this; }
CUDA_ALL
    EFP64& operator<<=(int64_t power) { eval = efp64_scale_power2(eval, power); return *this; }
CUDA_ALL
    EFP64& operator>>=(int64_t power) { eval = efp64_scale_power2(eval, -power); return *this; }
CUDA_ALL
    bool operator<(const EFP64 &other) const { return efp64_cmp(eval, other.eval) < 0; }
CUDA_ALL
    bool operator<=(const EFP64 &other) const { return efp64_cmp(eval, other.eval) <= 0; }
CUDA_ALL
    bool operator>(const EFP64 &other) const { return efp64_cmp(eval, other.eval) > 0; }
CUDA_ALL
    bool operator>=(const EFP64 &other) const { return efp64_cmp(eval, other.eval) >= 0; }

CUDA_ALL
    explicit operator unsigned() const { return efp64_to_unsigned(eval); }
CUDA_ALL
    explicit operator int() const { return efp64_to_int(eval); }
CUDA_ALL
    explicit operator fp64_t() const { return efp64_to_fp64(eval); }

/* Low-level operations to support SOA and fast multiplication */
CUDA_ALL
    void quick_mul_accum(const EFP64 &other) { eval = efp64_quick_mul(eval, other.eval); }

CUDA_ALL
    EFP64 canonicalize() { eval = efp64_canonicalize(eval); return *this;}

CUDA_ALL
    void disperse(fp64_t *dptr, int64_t *eptr) { efp64_disperse(eval, dptr, eptr); }

CUDA_ALL
    EFP64(fp64_t d, int64_t e) { eval = efp64_collect(d, e); }

/* Support for reduction operation */

CUDA_HOST
    friend EFP64 product_reduce_slow(EFP64 ival, EFP64 *data, int len) {
	efp64_t prod = ival.get_efp64_t();
	for (int i = 0; i < len; i++)
	    prod = efp64_mul(prod, data[i].get_efp64_t());
	return EFP64(prod);
    }

CUDA_HOST
    friend EFP64 product_reduce_x1(EFP64 ival, EFP64 *data, int len) {
	efp64_t prod = ival.get_efp64_t();
	int rcount = 0;
	for (int i = 0; i < len; i++) {
	    prod = efp64_quick_mul(prod, data[i].get_efp64_t());
	    if (++rcount > EFP64_MAX_MUL) {
		prod = efp64_canonicalize(prod);
		rcount = 0;
	    }
	}
	prod = efp64_canonicalize(prod);
	return EFP64(prod);
    }

CUDA_HOST
    friend EFP64 product_reduce_x4(EFP64 ival, EFP64 *data, int len) {
	// Assume len >= 4
	efp64_t prod[4];
	int i, j;
	for (j = 0; j < 4; j++) 
	    prod[j] = data[j].get_efp64_t();
	prod[0] = efp64_quick_mul(prod[0], ival.get_efp64_t());
	int rcount = 0;
	for (i = 4; i <= len-4; i+= 4) {
	    for (j = 0; j < 4; j++)
		prod[j] = efp64_quick_mul(prod[j], data[i+j].get_efp64_t());
	    if (++rcount > EFP64_MAX_MUL) {
		rcount = 0;
		for (j = 0; j < 4; j++)
		    prod[j] = efp64_canonicalize(prod[j]);
	    }
	}
	if (rcount * 4 > EFP64_MAX_MUL) {
	    for (j = 0; j < 4; j++)
		prod[j] = efp64_canonicalize(prod[j]);
	}
	efp64_t result = prod[0];
	for (j = 1; j < 4; j++)
	    result = efp64_quick_mul(result, prod[j]);
	for (; i < len; i++)
	    result = efp64_quick_mul(result, data[i].get_efp64_t());
	return EFP64(efp64_canonicalize(result));
    }


CUDA_HOST
    friend EFP64 product_reduce(EFP64 ival, EFP64 *data, int len) {
	if (len >= 8)
	    return product_reduce_x4(ival, data, len);
	else
	    return product_reduce_x1(ival, data, len);
    }

CUDA_HOST
    friend EFP64 product_reduce_slow(EFP64 ival, std::vector<EFP64> &data)
    { return product_reduce_slow(ival, data.data(), (int) data.size()); }

CUDA_HOST
    friend EFP64 product_reduce_x1(EFP64 ival, std::vector<EFP64> &data)
    { return product_reduce_x1(ival, data.data(), (int) data.size()); }

CUDA_HOST
    friend EFP64 product_reduce(EFP64 ival, std::vector<EFP64> &data)
    { return product_reduce(ival, data.data(), (int) data.size()); }



#if EFP64_NO_GMP
CUDA_HOST
    friend std::ostream& operator<<(std::ostream& os, const EFP64 &a) {
	char buf[EFP64_BUF];
	efp64_string(a.eval, buf, EFP64_NSIG);
	os << (const char *) buf;
	return os;
    }

CUDA_HOST
    friend std::istream& operator>>(std::istream& is, EFP64 &a) {
	std::string s;
	is >> s;
	if (efp64_sscanf(s.data(), &a.eval) == 0)
	    is.setstate(std::ios::failbit);
	return is;
    }
#else /* EFP64_NO_GMP */
    /* Use MPF for I/O */
CUDA_HOST
    friend std::ostream& operator<<(std::ostream& os, const EFP64 &a) {
	mpf_class ma(a.get_mpf(), 64);
	char buf[EFP64_BUF];
	mpf_string(buf, ma.get_mpf_t(), EFP64_NSIG);
	return (os << buf);
    }

CUDA_HOST
    friend std::istream& operator>>(std::istream& is, EFP64 &a) {
	std::string s;
	if (is >> s) {
	    mpf_class ma(s, 64);
	    a = ma;
	}
	return is;
    }
#endif /* EFP64_NO_GMP */
};
