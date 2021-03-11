set(ZHETAPI_SOURCE 
	source/barn.cpp
	source/complex.cpp
	source/display.cpp
	source/function.cpp
	source/image.cpp
	source/registration.cpp
	source/variable.cpp
	source/token.cpp

	source/core/algorithm.cpp
	source/core/class.cpp
	source/core/engine_base.cpp
	source/core/label.cpp
	source/core/lvalue.cpp
	source/core/node.cpp
	source/core/node_differential.cpp
	source/core/node_differentiation.cpp
	source/core/node_list.cpp
	source/core/node_manager.cpp
	source/core/node_reference.cpp
	source/core/operation.cpp
	source/core/operation_holder.cpp
	source/core/parser.cpp
	source/core/rvalue.cpp
	source/core/shader.cpp
	source/core/types.cpp
	source/core/variable_cluster.cpp

	source/std/functions.cpp
	source/std/interval.cpp
	source/std/linalg.cpp
	
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
