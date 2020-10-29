cmake_minimum_required(VERSION 3.14)

project(zhetapi)

set(CMAKE_CXX_COMPILER "g++-8")

add_executable(zhetapi cli/cli.cpp)
target_include_directories(zhetapi PUBLIC engine)
target_include_directories(zhetapi PUBLIC inc)
target_include_directories(zhetapi PUBLIC /usr/local/include)
