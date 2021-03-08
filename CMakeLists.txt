# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Settings
include(cmake/colors.cmake)
include(cmake/settings.cmake)

# Projects
project(zhetapi)
project(exp)
project(gui)
project(ml)
project(cuda)
project(port)

# Source for projects
include(cmake/zhetapi.cmake)
include(cmake/deps.cmake)
include(cmake/libs.cmake)
include(cmake/exp.cmake)
include(cmake/gui.cmake)
include(cmake/ml.cmake)
include(cmake/cuda.cmake)
include(cmake/port.cmake)
include(cmake/dirs.cmake)
