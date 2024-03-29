cmake_minimum_required(VERSION 3.12)

# All projects
project(zhetapi CXX)

# Check existence of CUDA
option(ZHETAPI_ENABLE_CUDA "Enable CUDA for Zhetapi" ON)

if (ZHETAPI_ENABLE_CUDA)
	enable_language(CUDA)
endif()

# CXX options
set(CMAKE_CXX_STANDARD 20)

# Compiler (clang default)
if (NOT CMAKE_CXX_COMPILER)
	set(CMAKE_CXX_COMPILER clang++)
endif()

# Color output
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")

# Build type (Release default)
if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Required packages
find_package(PNG REQUIRED)
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
	.
	vendor
	vendor/indicators/include
	vendor/benchmark/include
	${PNG_INCLUDE_DIRS}
)

# Zhetapi library sources
set(Zhetapi_SOURCES
	source/autograd/autograd.cpp
	source/autograd/iseq.cpp
	source/autograd/ml.cpp
	source/image.cpp
	source/io.cpp
	source/linalg.cpp
	source/polynomial.cpp
	source/range.cpp
	source/std/functions.cpp
	source/std/interval.cpp
	source/timer.cpp
)

# Common object library
add_library(Zhetapi_COMMON OBJECT ${Zhetapi_SOURCES})

# Project executables
add_executable(mnist experimental/mnist.cpp $<TARGET_OBJECTS:Zhetapi_COMMON>)
add_executable(experimental experimental/experimental.cpp $<TARGET_OBJECTS:Zhetapi_COMMON>)

if (ZHETAPI_ENABLE_CUDA)
        # add_executable(experimental_cuda experimental/cuda.cu ${Zhetapi_SOURCES})
endif()

add_executable(benchmark experimental/benchmark.cpp $<TARGET_OBJECTS:Zhetapi_COMMON>)

set(ESSENTIAL_LIBS PNG::PNG OpenMP::OpenMP_CXX)

target_link_libraries(mnist ${ESSENTIAL_LIBS})
target_link_libraries(experimental ${ESSENTIAL_LIBS})
# target_link_libraries(experimental_cuda ${ESSENTIAL_LIBS} cuda cudart nvrtc)
target_link_libraries(benchmark benchmark::benchmark ${ESSENTIAL_LIBS})

if (ZHETAPI_ENABLE_CUDA)
        target_link_libraries(experimental ${ESSENTIAL_LIBS})
endif()
