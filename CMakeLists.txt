# Set CMake version
cmake_minimum_required(VERSION 3.14)

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

add_executable(port source/port/port.cpp source/port/port-vector.cpp source/port/port-matrix.cpp source/port/port-tensor.cpp)

target_link_libraries(port PUBLIC zhp)

# Include directories
include_directories(PUBLIC engine)

include_directories(PUBLIC /usr/local/include)
