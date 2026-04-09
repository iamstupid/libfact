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

add_library(zfactor_fixint INTERFACE)
add_library(zfactor::fixint ALIAS zfactor_fixint)
target_include_directories(zfactor_fixint INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_library(zfactor_fixint_backend STATIC
        src/fixint/sqr_diag_addlsh1_x86_64.S)
    target_compile_options(zfactor_fixint_backend PRIVATE -march=native)
    target_link_libraries(zfactor_fixint INTERFACE zfactor_fixint_backend)
endif()
if(ZFACTOR_CPU_HAS_BMI2)
    target_compile_definitions(zfactor_fixint INTERFACE ZFACTOR_HAS_BMI2=1)
endif()
if(ZFACTOR_CPU_HAS_ADX)
    target_compile_definitions(zfactor_fixint INTERFACE ZFACTOR_HAS_ADX=1)
endif()

function(zfactor_apply_common_flags target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -march=native)
    target_compile_options(${target} PRIVATE $<$<CONFIG:Release>:-O3>)
endfunction()

add_executable(zfactor src/main.cpp)
target_include_directories(zfactor PRIVATE src)
target_link_libraries(zfactor PRIVATE zfactor_fixint)

if(NOT EMSCRIPTEN)
    target_link_libraries(zfactor PRIVATE mimalloc-static)
endif()

zfactor_apply_common_flags(zfactor)

# --- tests ---
if(ZFACTOR_BUILD_TESTS)
    enable_testing()

    add_executable(test_primesieve tests/primesieve/test_primesieve.cpp)
    target_include_directories(test_primesieve PRIVATE src third_party)
    if(NOT EMSCRIPTEN)
        target_link_libraries(test_primesieve PRIVATE mimalloc-static)
    endif()
    zfactor_apply_common_flags(test_primesieve)
    add_test(NAME primesieve COMMAND test_primesieve)

    add_executable(test_uint tests/fixint/test_uint.cpp)
    target_link_libraries(test_uint PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(test_uint)
    add_test(NAME uint COMMAND test_uint)
endif()

if(ZFACTOR_BUILD_BENCHMARKS)
    add_executable(bench_uint bench/fixint/bench_uint.cpp)
    target_link_libraries(bench_uint PRIVATE zfactor_fixint)
    zfactor_apply_common_flags(bench_uint)

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
