#pragma once

#include <gmp.h>
#include <stdint.h>

template< typename digit_t >
digit_t get_invmod(const uint8_t N[], int nbytes);

template< typename digit_t >
void get_R_and_Rsqr_mod(digit_t R_mod[], digit_t R_sqr_mod[], const uint8_t mod[], int nbytes);

#include "gmp_utils.cc"
