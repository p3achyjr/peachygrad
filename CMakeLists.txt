cmake_minimum_required(VERSION 3.5...3.26)
project(peachygrad LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR}/cc)

add_subdirectory(extern/pybind11)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
add_compile_options(-mavx2 -mfma)

# Add Sources.
file(GLOB SOURCES "cc/*.cc" "cc/*.h")
list(FILTER SOURCES EXCLUDE REGEX "bench\\.cc$")
pybind11_add_module(peachygrad_cc ${SOURCES})

file(GLOB BENCH_SOURCES "cc/*.cc" "cc/*.h")
list(FILTER SOURCES EXCLUDE REGEX "(peachyffi|api)\\.(cc|h)")
add_executable(bench ${BENCH_SOURCES})
target_link_libraries(bench PRIVATE pybind11::embed)
pybind11_add_module(bench_module SHARED ${BENCH_SOURCES})
