cmake_minimum_required(VERSION 3.15)
project(calculator)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(calculator main.cpp tokens.h)
