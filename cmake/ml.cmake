# Compile exp file
add_executable(ml experimental/ml.cpp)
target_link_libraries(ml ${PNG_LIBRARY} zhp-shared)