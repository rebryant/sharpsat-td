#pragma once
#ifdef __CUDACC__
#ifndef CUDA_ALL
#define CUDA_ALL __host__ __device__
#define CUDA_HOST __host__
#endif
#else
#define CUDA_ALL
#define CUDA_HOST
#endif

#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

CUDA_HOST
static void mpf_string(char *buf, mpf_srcptr val, int nsig) {
    int boffset = 0;
    mp_exp_t ecount;
    if (nsig <= 0)
	nsig = 1;
    if (nsig > 20)
	nsig = 20;
    char *sval = mpf_get_str(NULL, &ecount, 10, nsig, val);
    if (!sval || strlen(sval) == 0 || sval[0] == '0') {
	strcpy(buf, "0.0");
    } else {
	int voffset = 0;
	bool neg = sval[0] == '-';
	if (neg) {
	    voffset++;
	    buf[boffset++] = '-';
	}
	if (ecount == 0) {
	    buf[boffset++] = '0';
	    buf[boffset++] = '.';
	} else {
	    buf[boffset++] = sval[voffset++];
	    buf[boffset++] = '.';
	    ecount--;
	}
	if (sval[voffset] == 0)
	    buf[boffset++] = '0';
	else {
	    while(sval[voffset] != 0)
		buf[boffset++] = sval[voffset++];
	}
	if (ecount != 0) {
	    buf[boffset++] = 'e';
	    if (ecount > 0)
		buf[boffset++] = '+';
	    snprintf(&buf[boffset], 24, "%ld", (long) ecount);
	} else
	    buf[boffset] = 0;
    }
    free(sval);
}

