cmake_minimum_required(VERSION 3.20)
project(zfactor LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(FATAL_ERROR "MSVC is not supported. Use GCC or Clang, preferably under WSL.")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR "Unsupported compiler '${CMAKE_CXX_COMPILER_ID}'. Use GCC or Clang.")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "GCC 14 or newer is required. The primary target toolchain is g++-14 under WSL.")
endif()

option(ZFACTOR_BUILD_TESTS "Build zfactor tests" ON)
option(ZFACTOR_BUILD_BENCHMARKS "Build zfactor benchmarks" ON)
option(ZFACTOR_BENCH_GMP "Build bench_uint with GMP" OFF)
set(ZFACTOR_GMP_ROOT "" CACHE PATH "Optional GMP install prefix for benchmark builds")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(CheckCXXSourceRuns)

set(ZFACTOR_MPN_GENERATED "${CMAKE_CURRENT_SOURCE_DIR}/include/zfactor/fixint/detail/mpn_generated.h")
set(ZFACTOR_MONTMUL_ADCX_GENERATED "${CMAKE_CURRENT_SOURCE_DIR}/include/zfactor/fixint/detail/montmul_adcx_generated.h")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/zfactor/fixint/detail")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixint/gen_mpn.py"
            --output "${ZFACTOR_MPN_GENERATED}"
            --arch x86_64
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE ZFACTOR_MPN_GEN_RESULT
    )
    if(NOT ZFACTOR_MPN_GEN_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate mpn_generated.h")
    endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/fixint/gen_montmul_adcx.py"
            --output "${ZFACTOR_MONTMUL_ADCX_GENERATED}"
            --min-n 3 --max-n 6
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE ZFACTOR_MONTMUL_ADCX_GEN_RESULT
    )
    if(NOT ZFACTOR_MONTMUL_ADCX_GEN_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate montmul_adcx_generated.h")
    endif()

    check_cxx_source_runs("
        int main() {
            return __builtin_cpu_supports(\"bmi2\") ? 0 : 1;
        }" ZFACTOR_CPU_HAS_BMI2)
    check_cxx_source_runs("
        int main() {
            return __builtin_cpu_supports(\"adx\") ? 0 : 1;
        }" ZFACTOR_CPU_HAS_ADX)
endif()

if(NOT EMSCRIPTEN)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/mimalloc)
endif()

# --- primesieve (Kim Walisch) ---
set(BUILD_PRIMESIEVE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_MANPAGE OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/primesieve)

# --- zint (header-only bigint with NTT mul, by iamstupid) ---
add_subdirectory(third_party/zint)

add_library(zfactor_fixint INTERFACE)
add_library(zfactor::fixint ALIAS zfactor_fixint)
target_include_directories(zfactor_fixint INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# zfactor_sieve — header-only wrapper around third-party primesieve.
add_library(zfactor_sieve INTERFACE)
add_library(zfactor::sieve ALIAS zfactor_sieve)
target_include_directories(zfactor_sieve INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(zfactor_sieve INTERFACE primesieve::primesieve)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_library(zfactor_fixint_backend STATIC
        src/fixint/sqr_diag_addlsh1_x86_64.S
        src/fixint/montmul_x86_64.S
        src/fixint/divrem_x86_64.S
        src/fixint/divexact_x86_64.S)
    target_compile_options(zfactor_fixint_backend PRIVATE -march=native)
    target_link_libraries(zfactor_fixint INTERFACE zfactor_fixint_backend)
endif()
if(ZFACTOR_CPU_HAS_BMI2)
    target_compile_definitions(zfactor_fixint INTERFACE ZFACTOR_HAS_BMI2=1)
endif()
if(ZFACTOR_CPU_HAS_ADX)
    target_compile_definitions(zfactor_fixint INTERFACE ZFACTOR_HAS_ADX=1)
endif()
if(ZFACTOR_CPU_HAS_BMI2 AND ZFACTOR_CPU_HAS_ADX)
    target_compile_definitions(zfactor_fixint INTERFACE ZFACTOR_HAS_MONTMUL_IR=1)
endif()

# --- GMP (optional, for faster gcd/jacobi/modinv) ---
find_library(GMP_LIBRARY gmp)
if(GMP_LIBRARY)
    target_compile_definitions(zfactor_fixint INTERFACE ZFACTOR_USE_GMP=1)
    target_link_libraries(zfactor_fixint INTERFACE ${GMP_LIBRARY})
endif()


function(zfactor_apply_common_flags target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -march=native)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O3>)
endfunction()

add_executable(zfactor src/main.cpp)
target_include_directories(zfactor PRIVATE src)
target_link_libraries(zfactor PRIVATE zfactor_fixint zfactor_sieve)

if(NOT EMSCRIPTEN)
    target_link_libraries(zfactor PRIVATE mimalloc-static)
endif()

zfactor_apply_common_flags(zfactor)

# --- tests ---
if(ZFACTOR_BUILD_TESTS)
    enable_testing()

    add_executable(test_primesieve tests/primesieve/test_primesieve.cpp)
    target_include_directories(test_primesieve PRIVATE third_party)
    target_link_libraries(test_primesieve PRIVATE zfactor_sieve)
    if(NOT EMSCRIPTEN)
        target_link_libraries(test_primesieve PRIVATE mimalloc-static)
    endif()
    zfactor_apply_common_flags(test_primesieve)
    add_test(NAME primesieve COMMAND test_primesieve)

    add_executable(test_uint tests/fixint/test_uint.cpp)
    target_link_libraries(test_uint PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_uint)
    add_test(NAME uint COMMAND test_uint)

    add_executable(test_montgomery tests/fixint/test_montgomery.cpp)
    target_link_libraries(test_montgomery PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_montgomery)
    add_test(NAME montgomery COMMAND test_montgomery)

    add_executable(test_millerrabin tests/fixint/test_millerrabin.cpp)
    target_link_libraries(test_millerrabin PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_millerrabin)
    add_test(NAME millerrabin COMMAND test_millerrabin)

    add_executable(test_gcd tests/fixint/test_gcd.cpp)
    target_link_libraries(test_gcd PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_gcd)
    add_test(NAME gcd COMMAND test_gcd)

    add_executable(test_trial tests/fixint/test_trial.cpp)
    target_link_libraries(test_trial PRIVATE zfactor_fixint zfactor_sieve)
    target_include_directories(test_trial PRIVATE third_party)
    zfactor_apply_common_flags(test_trial)
    add_test(NAME trial COMMAND test_trial)

    add_executable(test_iroot tests/fixint/test_iroot.cpp)
    target_link_libraries(test_iroot PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_iroot)
    add_test(NAME iroot COMMAND test_iroot)

    add_executable(test_perfect_power tests/fixint/test_perfect_power.cpp)
    target_link_libraries(test_perfect_power PRIVATE zfactor_fixint zfactor_sieve)
    target_include_directories(test_perfect_power PRIVATE third_party)
    zfactor_apply_common_flags(test_perfect_power)
    add_test(NAME perfect_power COMMAND test_perfect_power)

    add_executable(test_qr tests/fixint/test_qr.cpp)
    target_link_libraries(test_qr PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_qr)
    add_test(NAME qr COMMAND test_qr)

    add_executable(test_bpsw tests/fixint/test_bpsw.cpp)
    target_link_libraries(test_bpsw PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_bpsw)
    add_test(NAME bpsw COMMAND test_bpsw)

    add_executable(test_rho tests/fixint/test_rho.cpp)
    target_link_libraries(test_rho PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_rho)
    add_test(NAME rho COMMAND test_rho)

    add_executable(test_squfof tests/fixint/test_squfof.cpp)
    target_link_libraries(test_squfof PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_squfof)
    add_test(NAME squfof COMMAND test_squfof)

    add_executable(test_divexact1 tests/fixint/test_divexact1.cpp)
    target_link_libraries(test_divexact1 PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_divexact1)
    add_test(NAME divexact1 COMMAND test_divexact1)

    add_executable(test_simd_mod tests/fixint/test_simd_mod.cpp)
    target_link_libraries(test_simd_mod PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_simd_mod)
    add_test(NAME simd_mod COMMAND test_simd_mod)

    # --- SIQS ---
    add_library(zfactor_lanczos STATIC src/siqs/lanczos.cpp)
    target_include_directories(zfactor_lanczos PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_options(zfactor_lanczos PRIVATE -Wall -Wextra -march=native -O2
        -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable)

    add_executable(test_siqs_debug tests/fixint/test_siqs_debug.cpp)
    target_link_libraries(test_siqs_debug PRIVATE zfactor_fixint zfactor_sieve)
    target_include_directories(test_siqs_debug PRIVATE src)
    zfactor_apply_common_flags(test_siqs_debug)

    add_executable(test_siqs tests/fixint/test_siqs.cpp)
    target_link_libraries(test_siqs PRIVATE zfactor_fixint zfactor_sieve zfactor_lanczos)
    target_include_directories(test_siqs PRIVATE src)
    zfactor_apply_common_flags(test_siqs)
    add_test(NAME siqs COMMAND test_siqs)
endif()

if(ZFACTOR_BUILD_BENCHMARKS)
    add_executable(bench_divexact bench/fixint/bench_divexact.cpp)
    target_link_libraries(bench_divexact PRIVATE zfactor_fixint zfactor_sieve)
    target_include_directories(bench_divexact PRIVATE third_party)
    zfactor_apply_common_flags(bench_divexact)

    add_executable(bench_trial_smooth bench/fixint/bench_trial_smooth.cpp)
    target_link_libraries(bench_trial_smooth PRIVATE zfactor_fixint zfactor_sieve)
    target_include_directories(bench_trial_smooth PRIVATE third_party)
    zfactor_apply_common_flags(bench_trial_smooth)

    add_executable(bench_simd_mod bench/fixint/bench_simd_mod.cpp)
    target_link_libraries(bench_simd_mod PRIVATE zfactor_fixint zfactor_sieve)
    target_include_directories(bench_simd_mod PRIVATE third_party)
    zfactor_apply_common_flags(bench_simd_mod)

    add_executable(bench_uint bench/fixint/bench_uint.cpp)
    target_link_libraries(bench_uint PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_uint)

    add_executable(bench_montgomery bench/fixint/bench_montgomery.cpp)
    target_link_libraries(bench_montgomery PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_montgomery)

    add_executable(bench_barrett bench/fixint/bench_barrett.cpp)
    target_link_libraries(bench_barrett PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_barrett)

    add_executable(bench_trial bench/fixint/bench_trial.cpp)
    target_link_libraries(bench_trial PRIVATE zfactor_fixint zfactor_sieve)
    target_include_directories(bench_trial PRIVATE third_party)
    zfactor_apply_common_flags(bench_trial)

    add_executable(bench_rho_vs_squfof bench/fixint/bench_rho_vs_squfof.cpp)
    target_link_libraries(bench_rho_vs_squfof PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_rho_vs_squfof)

    add_executable(bench_rho_inner bench/fixint/bench_rho_inner.cpp)
    target_link_libraries(bench_rho_inner PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_rho_inner)

    add_executable(bench_mpfq bench/fixint/bench_mpfq.cpp)
    target_link_libraries(bench_mpfq PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_mpfq)

    add_executable(test_zint_smoke bench/fixint/test_zint_smoke.cpp)
    target_link_libraries(test_zint_smoke PRIVATE zint)
    zfactor_apply_common_flags(test_zint_smoke)

    add_executable(test_eecm_setup bench/fixint/test_eecm_setup.cpp)
    target_link_libraries(test_eecm_setup PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_eecm_setup)

    add_executable(test_eecm_e2e bench/fixint/test_eecm_e2e.cpp)
    target_link_libraries(test_eecm_e2e PRIVATE zfactor_fixint zfactor_sieve zint)
    zfactor_apply_common_flags(test_eecm_e2e)

    # test_eecm_smul and test_eecm_stages use the old wNAF API (pre-accumulator).
    # Kept as source documentation but excluded from build.
    # add_executable(test_eecm_smul bench/fixint/test_eecm_smul.cpp)
    # add_executable(test_eecm_stages bench/fixint/test_eecm_stages.cpp)

    # bench_eecm uses old ecm<N>(n, factor_bits, max_curves) API.
    # add_executable(bench_eecm bench/fixint/bench_eecm.cpp)

    add_executable(bench_edwards bench/fixint/bench_edwards.cpp)
    target_link_libraries(bench_edwards PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_edwards)

    add_executable(test_eecm_160bit bench/fixint/test_eecm_160bit.cpp)
    target_link_libraries(test_eecm_160bit PRIVATE zfactor_fixint zfactor_sieve zint)
    zfactor_apply_common_flags(test_eecm_160bit)

    if(ZFACTOR_BENCH_GMP)
        find_path(ZFACTOR_GMP_INCLUDE_DIR gmp.h
            HINTS "${ZFACTOR_GMP_ROOT}" "${ZFACTOR_GMP_ROOT}/include"
            PATH_SUFFIXES include)
        find_library(ZFACTOR_GMP_LIBRARY
            NAMES gmp libgmp libgmp.a
            HINTS
                "${ZFACTOR_GMP_ROOT}/.libs"
                "${ZFACTOR_GMP_ROOT}"
                "${ZFACTOR_GMP_ROOT}/lib"
                "${ZFACTOR_GMP_ROOT}/lib64"
            PATH_SUFFIXES lib lib64 .libs)
        if(NOT ZFACTOR_GMP_INCLUDE_DIR OR NOT ZFACTOR_GMP_LIBRARY)
            message(FATAL_ERROR
                "ZFACTOR_BENCH_GMP=ON but GMP was not found. "
                "Install libgmp-dev in WSL or build a local copy with tools/wsl/build_gmp.sh "
                "and set ZFACTOR_GMP_ROOT to its install prefix.")
        endif()
        target_compile_definitions(bench_uint PRIVATE ZFACTOR_BENCH_GMP=1)
        target_include_directories(bench_uint PRIVATE "${ZFACTOR_GMP_INCLUDE_DIR}")
        target_link_libraries(bench_uint PRIVATE "${ZFACTOR_GMP_LIBRARY}")
    endif()
endif()
