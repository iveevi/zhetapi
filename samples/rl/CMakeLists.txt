# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Set the right flags
set(CMAKE_CXX_FLAGS "-pthread -std=c++17")
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "$CMAKE_CXX_FLAGS -g")
endif ()

# Set compiler
set(CMAKE_CXX_COMPILER "g++-8")

# Project listing
project(rl)

# Required libraries (create a separate cmake file for this)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Main project
add_executable(rl
	agent.cpp
	field.cpp
	heurestic.cpp
	main.cpp
	vis.cpp
)

target_link_libraries(rl ${OPENGL_LIBRARIES})
target_link_libraries(rl ${GLUT_LIBRARY})

include_directories(../../engine)
