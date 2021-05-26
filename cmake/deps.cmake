# PNG
find_package(PNG QUIET)

if (PNG_FOUND)
	message("${present}Found PNG library.${reset}")
else()
	message("${notify}Missing OpenGL: GUI functions such as Image::show() will not be available.${reset}")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DZHP_NO_GUI=0")
endif()

include_directories(${PNG_INCLUDE_DIR})
