# Compile zhetapi file
add_executable(zhetapi
	interpreter/main.cpp
	interpreter/parser.cpp
	interpreter/library.cpp
	interpreter/execute.cpp
	interpreter/builtin/basic_io.cpp
)

target_link_libraries(zhetapi PUBLIC zhp-static)
target_link_libraries(zhetapi PUBLIC ${CMAKE_DL_LIBS})
