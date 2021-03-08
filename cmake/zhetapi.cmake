# Compile czhp file
add_executable(zhetapi
	czhp/czhp.cpp
	czhp/parser.cpp
	czhp/library.cpp
	czhp/execute.cpp
	czhp/builtin/basic_io.cpp
)

target_link_libraries(zhetapi PUBLIC zhp-static)
target_link_libraries(zhetapi PUBLIC ${CMAKE_DL_LIBS})
