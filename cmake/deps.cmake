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