cmake_minimum_required(VERSION 3.14)

if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "-g")
endif ()

project(mnist)

add_executable(mnist main.cpp)

include_directories(../../engine)
