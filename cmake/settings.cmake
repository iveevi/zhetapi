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