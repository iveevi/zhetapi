# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Colors
if(NOT WIN32)
	string(ASCII 27 esc)

	set(reset 	"${esc}[m")
	set(present	"${esc}[1;32m")
	set(notify	"${esc}[1;31m")
endif()

# Set the right flags
set(CMAKE_CXX_FLAGS "-pthread -std=c++17")
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "-pthread -g -std=c++17 -Wall")
endif ()

# Allow dynamic linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Set compiler
set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

# Projects
project(exp)

# Packages
find_package(PNG QUIET)

if (PNG_FOUND)
	message("${present}Found PNG library.${reset}")
else()
endif()

find_package(OpenGL QUIET)

if (OpenGL_FOUND)
	message("${present}Found OpenGL.${reset}")
else()
	message("${notify}Missing OpenGL: GUI functions such as Image::show() will not be available.${reset}")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DZHP_NO_GUI=0")
endif()

find_package(glfw3 QUIET)

set(GLFW_LIBRARY "")
if (glfw3_FOUND)
	set(GLFW_LIBRARY "glfw")

	message("${present}Found GLFW3.${reset}")
else()
	message("${notify}Missing glfw3: GUI functions such as Image::show() will not be available.${reset}")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DZHP_NO_GUI=0")
endif()

include_directories(${PNG_INCLUDE_DIR})

# set(EXT_SOURCE
# glad/glad.c
# )

set(ZHETAPI_SOURCE 
	source/algorithm.cpp
	source/barn.cpp
	source/class.cpp
	source/complex.cpp
	source/display.cpp
	source/function.cpp
	source/image.cpp
	source/label.cpp
	source/node.cpp
	source/node_differential.cpp
	source/node_differentiation.cpp
	source/node_manager.cpp
	source/node_reference.cpp
	source/operation.cpp
	source/operation_holder.cpp
	source/registration.cpp
	source/token.cpp
	source/types.cpp
	source/variable.cpp
	source/variable_cluster.cpp
	source/std_functions.cpp
	source/shader.cpp
	glad/glad.c
)

# Compile shared and static library
add_library(zhp-shared SHARED ${ZHETAPI_SOURCE})

target_link_libraries(zhp-shared ${PNG_LIBRARY})
target_link_libraries(zhp-shared ${CMAKE_DL_LIBS})
target_link_libraries(zhp-shared ${OPENGL_LIBRARIES})
target_link_libraries(zhp-shared ${GLFW_LIBRARY})

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
find_package(OpenCV QUIET)
add_executable(exp source/exp/exp.cpp ${EXT_SOURCE})

target_link_libraries(exp ${PNG_LIBRARY} zhp-shared ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Compile exp file
add_executable(gui source/exp/gui.cpp ${EXT_SOURCE})

target_link_libraries(gui ${PNG_LIBRARY} ${OPENGL_LIBRARIES} glfw zhp-shared)

include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

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
	endif()
endif()

# Compile czhp file
project(czhp)

add_executable(czhp
	source/czhp/czhp.cpp
	source/czhp/parser.cpp
	source/czhp/library.cpp
	source/czhp/execute.cpp
	source/builtin/basic_io.cpp
)

target_link_libraries(czhp PUBLIC zhp-static)

# Compile portability tests
project(port)

add_library(port_vector OBJECT source/port/port-vector.cpp)
add_library(port_matrix OBJECT source/port/port-matrix.cpp)
add_library(port_tensor OBJECT source/port/port-tensor.cpp)
add_library(port_function OBJECT source/port/port-function.cpp)
add_library(port_special OBJECT source/port/port-special.cpp)
add_library(port_calculus OBJECT source/port/port-calculus.cpp)
add_library(timer OBJECT source/port/timers.cpp)

add_executable(port source/port/port.cpp)

target_link_libraries(port PUBLIC zhp-shared)

target_link_libraries(port PUBLIC port_vector)
target_link_libraries(port PUBLIC port_matrix)
target_link_libraries(port PUBLIC port_tensor)
target_link_libraries(port PUBLIC port_function)
target_link_libraries(port PUBLIC port_special)
target_link_libraries(port PUBLIC port_calculus)

target_link_libraries(port PUBLIC timer)

target_link_libraries(port PUBLIC ${CMAKE_DL_LIBS})

target_link_libraries(czhp PUBLIC ${CMAKE_DL_LIBS})

# Include directories
include_directories(PUBLIC engine)
include_directories(PUBLIC glad)

include_directories(PUBLIC /usr/local/include)
include_directories(PUBLIC CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
