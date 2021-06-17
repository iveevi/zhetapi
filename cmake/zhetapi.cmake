# Compile zhetapi file
add_executable(zhetapi
	interpreter/main.cpp
	interpreter/library.cpp
	# TODO: Do we need to add these here are well?
	interpreter/builtin/io.cpp
	interpreter/builtin/utility.cpp
)

target_link_libraries(zhetapi PUBLIC zhp-static)
target_link_libraries(zhetapi PUBLIC ${CMAKE_DL_LIBS})
