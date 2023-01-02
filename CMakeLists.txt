project(mnist)
project(experimental)
project(experimental_cuda CXX CUDA)

# CXX options
set(CMAKE_CXX_STANDARD 17)

# Compiler (clang default)
if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
endif()

# Build type (Release default)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# PNG library
find_package(PNG REQUIRED)

# Include directories
include_directories(. vendor vendor/indicators/include)

# Zhetapi library sources
# TODO: turn into a library
set(Zhetapi_SOURCES
	source/autograd/activation.cpp
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

add_executable(mnist experimental/mnist.cpp ${Zhetapi_SOURCES})
add_executable(experimental experimental/experimental.cpp ${Zhetapi_SOURCES})
add_executable(experimental_cuda experimental/cuda.cu ${Zhetapi_SOURCES})

target_link_libraries(mnist PNG::PNG)
target_link_libraries(experimental PNG::PNG)
target_link_libraries(experimental_cuda PNG::PNG cuda cudart nvrtc)
