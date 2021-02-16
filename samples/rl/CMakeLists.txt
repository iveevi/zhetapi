# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Set the right flags
include(../../cmake/settings.cmake)

# Project listing
project(rl)

# Required libraries (create a separate cmake file for this)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Main project
add_executable(rl
	environment.cpp
	field.cpp
	main.cpp
)

target_link_libraries(rl ${OPENGL_LIBRARIES})
target_link_libraries(rl ${GLUT_LIBRARY})
target_link_libraries(rl ${CMAKE_DL_LIBS})

include_directories(../../engine)
