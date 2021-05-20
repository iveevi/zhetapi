# Compile portability tests
add_executable(port
	testing/port/port-activation.cpp
	testing/port/port-calculus.cpp
	testing/port/port-function.cpp
	testing/port/port-interval.cpp
	testing/port/port-linalg.cpp
	testing/port/port-matrix.cpp
	testing/port/port-module.cpp
	testing/port/port-special.cpp
	testing/port/port-tensor.cpp
	testing/port/port-vector.cpp
	testing/port/port-fourier.cpp
	testing/port/port-parsing.cpp
	testing/port/port-polynomial.cpp
	testing/port/port.cpp
	testing/port/printing.cpp
	testing/port/timers.cpp
)

target_link_libraries(port PUBLIC zhp-shared)
target_link_libraries(port PUBLIC ${CMAKE_DL_LIBS})
