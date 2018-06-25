#pragma once

#include <gmp.h>
#include <stdint.h>

template< typename digit_t >
unsigned long get_invmod(const digit_t N[], int nelts, int modbits);

template< typename digit_t >
void get_R_and_Rsqr_mod(digit_t R_mod[], digit_t R_sqr_mod[], const uint8_t mod[], int nbytes);

#include "gmp_utils.cc"
