# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Set the right flags
if (${CMAKE_BUILD_TYPE} MATCHES DEBUG)
	set (CMAKE_CXX_FLAGS "-g")
endif ()

# Allow dynamic linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Set compiler
set(CMAKE_CXX_COMPILER "g++-8")

# Compile shared library
add_library(zhp SHARED source/class.cpp source/complex.cpp source/label.cpp source/node_reference.cpp source/operation_holder.cpp source/operation.cpp source/variable_cluster.cpp source/token.cpp source/node.cpp)

# Compile CLI application
project(zhetapi)

add_executable(zhetapi source/cli/cli.cpp)

target_link_libraries(zhetapi PUBLIC zhp)

# Compile portability tests
project(port)

add_library(port_vector OBJECT source/port/port-vector.cpp)
add_library(port_matrix OBJECT source/port/port-matrix.cpp)
add_library(port_tensor OBJECT source/port/port-tensor.cpp)
add_library(port_function OBJECT source/port/port-function.cpp)
add_library(timer OBJECT source/port/timers.cpp)

add_executable(port source/port/port.cpp)

target_link_libraries(port PUBLIC zhp)

target_link_libraries(port PUBLIC port_vector)
target_link_libraries(port PUBLIC port_matrix)
target_link_libraries(port PUBLIC port_tensor)
target_link_libraries(port PUBLIC port_function)
target_link_libraries(port PUBLIC timer)

target_link_libraries(port PUBLIC ${CMAKE_DL_LIBS})

# Include directories
include_directories(PUBLIC engine)

include_directories(PUBLIC /usr/local/include)
