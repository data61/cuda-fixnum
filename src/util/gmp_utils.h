#ifndef GMP_UTILS_H
#define GMP_UTILS_H

#include <gmp.h>
#include <stdint.h>

template< typename digit_t >
digit_t get_invmod(const uint8_t N[], int nbytes);
void get_R_and_Rsqr_mod(uint8_t R_mod[], uint8_t R_sqr_mod[], const uint8_t mod[], int nbytes);

#endif /* GMP_UTILS_H */
