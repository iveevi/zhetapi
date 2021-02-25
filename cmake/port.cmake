# Compile portability tests
add_library(port_vector OBJECT source/port/port-vector.cpp)
add_library(port_matrix OBJECT source/port/port-matrix.cpp)
add_library(port_tensor OBJECT source/port/port-tensor.cpp)
add_library(port_function OBJECT source/port/port-function.cpp)
add_library(port_special OBJECT source/port/port-special.cpp)
add_library(port_calculus OBJECT source/port/port-calculus.cpp)
add_library(port_interval OBJECT source/port/port-interval.cpp)
add_library(timer OBJECT source/port/timers.cpp)

add_executable(port source/port/port.cpp)

target_link_libraries(port PUBLIC zhp-shared)

target_link_libraries(port PUBLIC port_vector)
target_link_libraries(port PUBLIC port_matrix)
target_link_libraries(port PUBLIC port_tensor)
target_link_libraries(port PUBLIC port_function)
target_link_libraries(port PUBLIC port_special)
target_link_libraries(port PUBLIC port_calculus)
target_link_libraries(port PUBLIC port_interval)

target_link_libraries(port PUBLIC timer)

target_link_libraries(port PUBLIC ${CMAKE_DL_LIBS})

target_link_libraries(czhp PUBLIC ${CMAKE_DL_LIBS})