# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Set the right flags
set(CMAKE_CXX_FLAGS "-pthread -std=c++17")
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "$CMAKE_CXX_FLAGS -g")
endif ()

# Set compiler
set(CMAKE_CXX_COMPILER "g++-8")

# Main project
project(rl)

add_executable(rl main.cpp)

include_directories(../../engine)