# Compile CLI application
add_executable(zhetapi source/cli/cli.cpp)

target_link_libraries(zhetapi PUBLIC zhp-shared)