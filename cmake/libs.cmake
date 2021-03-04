set(ZHETAPI_SOURCE 
	source/algorithm.cpp
	source/barn.cpp
	source/class.cpp
	source/complex.cpp
	source/display.cpp
	source/function.cpp
	source/image.cpp
	source/interval.cpp
	source/label.cpp
	source/lvalue.cpp
	source/node.cpp
	source/node_differential.cpp
	source/node_differentiation.cpp
	source/node_manager.cpp
	source/node_reference.cpp
	source/operation.cpp
	source/operation_holder.cpp
	source/registration.cpp
	source/token.cpp
	source/types.cpp
	source/variable.cpp
	source/variable_cluster.cpp
	source/std_functions.cpp
	source/shader.cpp
	glad/glad.c
)

# Compile shared and static library
add_library(zhp-shared SHARED ${ZHETAPI_SOURCE})

target_link_libraries(zhp-shared ${PNG_LIBRARY})
target_link_libraries(zhp-shared ${CMAKE_DL_LIBS})
target_link_libraries(zhp-shared ${OPENGL_LIBRARIES})
target_link_libraries(zhp-shared ${GLFW_LIBRARY})

SET_TARGET_PROPERTIES(zhp-shared PROPERTIES
   OUTPUT_NAME zhp CLEAN_DIRECT_OUTPUT 1)

add_library(zhp-static STATIC ${ZHETAPI_SOURCE})

SET_TARGET_PROPERTIES(zhp-static PROPERTIES
   OUTPUT_NAME zhp CLEAN_DIRECT_OUTPUT 1)