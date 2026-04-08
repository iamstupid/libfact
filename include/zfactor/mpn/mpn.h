#pragma once

#if defined(_MSC_VER) && defined(_M_X64)
#include "mpn_intrin.h"
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#include "mpn_generated.h"
#else
#include "mpn_generic.h"
#endif
