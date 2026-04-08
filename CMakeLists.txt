cmake_minimum_required(VERSION 3.20)
project(zfactor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ZFACTOR_BUILD_TESTS "Build zfactor tests" ON)
option(ZFACTOR_BUILD_BENCHMARKS "Build zfactor benchmarks" ON)
option(ZFACTOR_BENCH_GMP "Build bench_uint with GMP" OFF)
set(ZFACTOR_GMP_ROOT "C:/Users/zball/Documents/bigint/gmp-6.3.0" CACHE PATH "Root of GMP build/install")

find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(CheckCXXSourceRuns)

set(ZFACTOR_MPN_GENERATED "${CMAKE_CURRENT_SOURCE_DIR}/include/zfactor/mpn/mpn_generated.h")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/zfactor/mpn")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_mpn.py"
            --output "${ZFACTOR_MPN_GENERATED}"
            --max-n 16
            --arch x86_64
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE ZFACTOR_MPN_GEN_RESULT
    )
    if(NOT ZFACTOR_MPN_GEN_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to generate mpn_generated.h")
    endif()

    check_cxx_source_runs("
        #if defined(_MSC_VER)
        #include <intrin.h>
        #endif
        int main() {
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_cpu_supports(\"bmi2\") ? 0 : 1;
        #elif defined(_MSC_VER)
            int info[4] = {};
            __cpuidex(info, 7, 0);
            return (info[1] & (1 << 8)) ? 0 : 1;
        #else
            return 1;
        #endif
        }" ZFACTOR_CPU_HAS_BMI2)
    check_cxx_source_runs("
        #if defined(_MSC_VER)
        #include <intrin.h>
        #endif
        int main() {
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_cpu_supports(\"adx\") ? 0 : 1;
        #elif defined(_MSC_VER)
            int info[4] = {};
            __cpuidex(info, 7, 0);
            return (info[1] & (1 << 19)) ? 0 : 1;
        #else
            return 1;
        #endif
        }" ZFACTOR_CPU_HAS_ADX)
endif()

if(NOT EMSCRIPTEN)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/mimalloc)
endif()

add_library(zfactor_mpn INTERFACE)
target_include_directories(zfactor_mpn INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
if(ZFACTOR_CPU_HAS_BMI2)
    target_compile_definitions(zfactor_mpn INTERFACE ZFACTOR_HAS_BMI2=1)
endif()
if(ZFACTOR_CPU_HAS_ADX)
    target_compile_definitions(zfactor_mpn INTERFACE ZFACTOR_HAS_ADX=1)
endif()

function(zfactor_apply_common_flags target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /fp:fast /W4)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -march=native)
        target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O3>)
    endif()
endfunction()

add_executable(zfactor src/main.cpp)
target_include_directories(zfactor PRIVATE src)
target_link_libraries(zfactor PRIVATE zfactor_mpn)

if(NOT EMSCRIPTEN)
    target_link_libraries(zfactor PRIVATE mimalloc-static)
endif()

zfactor_apply_common_flags(zfactor)

# --- tests ---
if(ZFACTOR_BUILD_TESTS)
    enable_testing()

    add_executable(test_primesieve test/test_primesieve.cpp)
    target_include_directories(test_primesieve PRIVATE src third_party)
    if(NOT EMSCRIPTEN)
        target_link_libraries(test_primesieve PRIVATE mimalloc-static)
    endif()
    zfactor_apply_common_flags(test_primesieve)
    add_test(NAME primesieve COMMAND test_primesieve)

    add_executable(test_uint tests/test_uint.cpp)
    target_link_libraries(test_uint PRIVATE zfactor_mpn)
    zfactor_apply_common_flags(test_uint)
    add_test(NAME uint COMMAND test_uint)
endif()

if(ZFACTOR_BUILD_BENCHMARKS)
    add_executable(bench_uint tests/bench_uint.cpp)
    target_link_libraries(bench_uint PRIVATE zfactor_mpn)
    zfactor_apply_common_flags(bench_uint)

    if(ZFACTOR_BENCH_GMP)
        find_path(ZFACTOR_GMP_INCLUDE_DIR gmp.h
            HINTS "${ZFACTOR_GMP_ROOT}" "${ZFACTOR_GMP_ROOT}/include")
        find_library(ZFACTOR_GMP_LIBRARY
            NAMES gmp libgmp libgmp.a
            HINTS "${ZFACTOR_GMP_ROOT}/.libs" "${ZFACTOR_GMP_ROOT}" "${ZFACTOR_GMP_ROOT}/lib")
        if(NOT ZFACTOR_GMP_INCLUDE_DIR OR NOT ZFACTOR_GMP_LIBRARY)
            message(FATAL_ERROR "ZFACTOR_BENCH_GMP=ON but GMP was not found under ${ZFACTOR_GMP_ROOT}")
        endif()
        target_compile_definitions(bench_uint PRIVATE ZFACTOR_BENCH_GMP=1)
        target_include_directories(bench_uint PRIVATE "${ZFACTOR_GMP_INCLUDE_DIR}")
        target_link_libraries(bench_uint PRIVATE "${ZFACTOR_GMP_LIBRARY}")
    endif()
endif()
