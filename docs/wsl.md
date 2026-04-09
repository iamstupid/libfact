# WSL Development

The primary development environment is WSL with `g++-14`. Clang is still
available for cross-checking, but the main supported path is GNU C++ on Linux.

## Toolchain

Install the baseline toolchain inside WSL:

```bash
sudo apt-get update
sudo apt-get install -y g++-14 gcc-14 clang cmake ninja-build python3 curl m4
```

The provided CMake presets are meant to be run from inside WSL.

## Configure And Build

Primary GCC 14 release build:

```bash
cmake --preset linux-gcc14-release
cmake --build --preset linux-gcc14-release
ctest --preset linux-gcc14-release
```

Optional Clang release build:

```bash
cmake --preset linux-clang-release
cmake --build --preset linux-clang-release
ctest --preset linux-clang-release
```

## GMP For Benchmarks

`bench_uint` can link against GMP when `ZFACTOR_BENCH_GMP=ON`.

To build a local GMP copy under the repo:

```bash
chmod +x tools/wsl/build_gmp.sh
./tools/wsl/build_gmp.sh
```

This installs GMP under `build/deps/gmp-install`.

Then configure the GMP-enabled preset:

```bash
cmake --preset linux-gcc14-release-gmp
cmake --build --preset linux-gcc14-release-gmp --target bench_uint
```

If you prefer a system GMP install instead, install `libgmp-dev` and either:

- leave `ZFACTOR_GMP_ROOT` empty so CMake uses the normal system search paths
- or set `-DZFACTOR_GMP_ROOT=/your/prefix`
