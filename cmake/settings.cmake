# Set the right flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-pthread")
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "-pthread -g -Wall")
endif ()

# Allow dynamic linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Set compiler (checks in /usr/bin/ or /usr/local/bin/)
if (EXISTS "/usr/bin/g++-8")
	set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
else()
	set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-8")
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
