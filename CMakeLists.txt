cmake_minimum_required(VERSION 3.10)
project(calculator)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(calculator src/main.cpp)
