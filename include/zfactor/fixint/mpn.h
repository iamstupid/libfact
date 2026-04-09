#pragma once

#if !defined(__GNUC__) && !defined(__clang__)
#error "zfactor fixint mpn supports GCC/Clang only. Use GCC or Clang, preferably under WSL."
#endif

#if defined(__x86_64__)
#include "detail/mpn_generated.h"
#else
#include "detail/mpn_generic.h"
#endif
