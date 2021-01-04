# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Set the right flags
set(CMAKE_CXX_FLAGS "-pthread -std=c++17")
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "-pthread -g -std=c++17 -Wall")
endif ()

# Allow dynamic linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Set compiler
set(CMAKE_CXX_COMPILER "g++-8")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(ZHETAPI_SOURCE 
	source/class.cpp
	source/complex.cpp
	source/label.cpp
	source/node_reference.cpp
	source/operation_holder.cpp
	source/operation.cpp
	source/registration.cpp
	source/variable_cluster.cpp
	source/token.cpp
	source/node.cpp
	source/display.cpp
)

# Compile shared library
add_library(zhp-shared SHARED ${ZHETAPI_SOURCE})

SET_TARGET_PROPERTIES(zhp-shared PROPERTIES
   OUTPUT_NAME zhp CLEAN_DIRECT_OUTPUT 1)

add_library(zhp-static STATIC ${ZHETAPI_SOURCE})

SET_TARGET_PROPERTIES(zhp-static PROPERTIES
   OUTPUT_NAME zhp CLEAN_DIRECT_OUTPUT 1)

# Compile CLI application
project(zhetapi)

add_executable(zhetapi source/cli/cli.cpp)

target_link_libraries(zhetapi PUBLIC zhp-shared)

# Compile exp file
project(exp)

add_executable(exp source/exp/exp.cpp)

target_link_libraries(exp PUBLIC zhp-shared)

# Compile cuda program
find_package(CUDA)
if (CUDA_FOUND)
	project(cuda LANGUAGES CXX CUDA)
	
	add_executable(cuda source/cuda/main.cu)

	target_link_libraries(cuda PUBLIC zhp-shared)
	target_link_libraries(cuda PUBLIC ${CUDA_LIBRARIES})
	target_link_libraries(cuda PUBLIC cudart)

	set_property(TARGET cuda PROPERTY CUDA_ARCHITECTURES 70)


	if (${CMAKE_BUILD_TYPE} MATCHES Debug)
		target_compile_options(cuda PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -g -G>)
	else()
		target_compile_options(cuda PRIVATE
			$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -O3>)
	endif ()
endif ()

# Compile czhp file
project(czhp)

add_executable(czhp source/czhp/czhp.cpp source/czhp/builtin.cpp)

target_link_libraries(czhp PUBLIC zhp-static)

# Compile portability tests
project(port)

add_library(port_vector OBJECT source/port/port-vector.cpp)
add_library(port_matrix OBJECT source/port/port-matrix.cpp)
add_library(port_tensor OBJECT source/port/port-tensor.cpp)
add_library(port_function OBJECT source/port/port-function.cpp)
add_library(timer OBJECT source/port/timers.cpp)

add_executable(port source/port/port.cpp)

target_link_libraries(port PUBLIC zhp-shared)

target_link_libraries(port PUBLIC port_vector)
target_link_libraries(port PUBLIC port_matrix)
target_link_libraries(port PUBLIC port_tensor)
target_link_libraries(port PUBLIC port_function)
target_link_libraries(port PUBLIC timer)

target_link_libraries(port PUBLIC ${CMAKE_DL_LIBS})

# Include directories
include_directories(PUBLIC engine)
include_directories(PUBLIC json)

include_directories(PUBLIC /usr/local/include)
include_directories(PUBLIC CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
