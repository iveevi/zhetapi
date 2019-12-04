cmake_minimum_required(VERSION 3.15)
project(calculator)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_executable(calculator
        src/main.cpp
        src/tokens.h
        src/types.h)
