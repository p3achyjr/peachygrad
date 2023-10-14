cmake_minimum_required(VERSION 3.5...3.26)
project(peachygrad LANGUAGES CXX)

include_directories(${CMAKE_SOURCE_DIR}/cc)

add_subdirectory(extern/pybind11)

set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-mavx2)

# Add Sources.
file(GLOB SOURCES "cc/*.cc" "cc/*.h")
pybind11_add_module(peachygrad_cc ${SOURCES})
