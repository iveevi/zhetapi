# Compile portability tests
add_executable(port
	source/port/port-calculus.cpp
	source/port/port-function.cpp
	source/port/port-interval.cpp
	source/port/port-linalg.cpp
	source/port/port-matrix.cpp
	source/port/port-special.cpp
	source/port/port-tensor.cpp
	source/port/port-vector.cpp
	source/port/port.cpp
	source/port/timers.cpp)

target_link_libraries(port PUBLIC zhp-shared)
target_link_libraries(port PUBLIC ${CMAKE_DL_LIBS})
