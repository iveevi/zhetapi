# Set CMake version
cmake_minimum_required(VERSION 3.14)

# Settings
include(cmake/colors.cmake)
include(cmake/settings.cmake)

# Projects
project(zhetapi)
project(exp)
project(exp)
project(gui)
project(cuda)
project(czhp)
project(port)

# Source for projects
include(cmake/deps.cmake)
include(cmake/libs.cmake)
include(cmake/zhetapi.cmake)
include(cmake/exp.cmake)
include(cmake/gui.cmake)
include(cmake/cuda.cmake)
include(cmake/czhp.cmake)
include(cmake/port.cmake)
include(cmake/dirs.cmake)
