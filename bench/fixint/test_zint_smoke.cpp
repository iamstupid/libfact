// Smoke test: include zint, multiply two bignums, print result.
#include <zint/zint.hpp>
#include <cstdio>

int main() {
    zint::bigint a = 1, b = 1;
    for (int i = 0; i < 100; ++i) a *= 3;
    for (int i = 0; i < 100; ++i) b *= 5;
    zint::bigint c = a * b;
    std::printf("3^100 * 5^100 = %s\n", c.to_string().c_str());
    return 0;
}
