// C++ bridge: exposes zint::mpn_mul as a C-linkage function callable
// from the FLINT poly mul override.
#include <zint/zint.hpp>

extern "C"
void zint_mpn_mul_bridge(unsigned long long* rp,
                         const unsigned long long* ap, unsigned int an,
                         const unsigned long long* bp, unsigned int bn)
{
    zint::mpn_mul(
        reinterpret_cast<zint::limb_t*>(rp),
        reinterpret_cast<const zint::limb_t*>(ap), an,
        reinterpret_cast<const zint::limb_t*>(bp), bn);
}
