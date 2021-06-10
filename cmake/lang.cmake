# Compile exp file
add_executable(lang experimental/lang.cpp ${EXT_SOURCE})
target_link_libraries(lang zhp-shared)
