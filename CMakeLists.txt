cmake_minimum_required(VERSION 3.20)
project(zfactor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT EMSCRIPTEN)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/mimalloc)
endif()

add_executable(zfactor src/main.cpp)
target_include_directories(zfactor PRIVATE src)

if(NOT EMSCRIPTEN)
    target_link_libraries(zfactor PRIVATE mimalloc-static)
endif()

if(MSVC)
    target_compile_options(zfactor PRIVATE /O2 /fp:fast /W4)
else()
    target_compile_options(zfactor PRIVATE -O2 -Wall -Wextra -march=native)
endif()
