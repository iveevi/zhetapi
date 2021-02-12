# Compile czhp file
add_executable(czhp
	source/czhp/czhp.cpp
	source/czhp/parser.cpp
	source/czhp/library.cpp
	source/czhp/execute.cpp
	source/builtin/basic_io.cpp
)

target_link_libraries(czhp PUBLIC zhp-static)