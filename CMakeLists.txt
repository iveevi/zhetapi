cmake_minimum_required(VERSION 3.14)

project(zhetapi)

set(CMAKE_CXX_COMPILER "g++-8")

add_executable(zhetapi source/cli.cpp)

include_directories(PUBLIC engine)
include_directories(PUBLIC inc)
include_directories(PUBLIC /usr/local/include)
