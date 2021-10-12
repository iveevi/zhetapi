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
# TODO: Make a list of compilers? Or do in run.py
if (EXISTS "/usr/bin/clang++-10")
	message("${warning}Using CLANG compiler.${reset}")
	set(CMAKE_CXX_COMPILER "/usr/bin/clang++-10")
elseif (EXISTS "/usr/bin/g++-8")
	message("${warning}Using GCC compiler.${reset}")
	set(CMAKE_CXX_COMPILER "/usr/bin/g++-8")
else()
	message("Requiring GCC compiler.")
	set(CMAKE_CXX_COMPILER "${warning}/usr/local/bin/g++-8${reset}")
endif()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
