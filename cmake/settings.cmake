# Set the right flags
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "-pthread")
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
elseif (${CMAKE_BUILD_TYPE} MATCHES Warn)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif (${CMAKE_BUILD_TYPE} MATCHES Codecov)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
endif ()

# Replace extensions
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)

# Allow dynamic linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

# Set compiler (checks in /usr/bin/ or /usr/local/bin/)
if (EXISTS "/usr/bin/g++-8")
	set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
else()
	set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-8")
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
