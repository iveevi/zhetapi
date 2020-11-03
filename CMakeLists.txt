cmake_minimum_required(VERSION 3.14)

project(zhetapi)
project(exp)
project(portability)

set(CMAKE_CXX_COMPILER "g++-8")

add_executable(zhetapi cli/cli.cpp)
add_executable(exp tests/tests.cpp)
add_executable(portability tests/portability.cpp)

include_directories(PUBLIC engine)
include_directories(PUBLIC inc)
include_directories(PUBLIC /usr/local/include)
