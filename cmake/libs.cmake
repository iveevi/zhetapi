set(ZHETAPI_SOURCE 
	source/complex.cpp
	source/display.cpp
	source/engine.cpp
	source/equation.cpp
	source/function.cpp
	source/image.cpp
	source/operand.cpp
	source/plot.cpp
	source/polynomial.cpp
	source/registration.cpp
	source/token.cpp

	source/core/algorithm.cpp
	source/core/class.cpp
	source/core/collection.cpp
	source/core/common.cpp
	source/core/engine_base.cpp
	source/core/expr_parser.cpp
	source/core/label.cpp
	source/core/lvalue.cpp
	source/core/module.cpp
	source/core/node.cpp
	source/core/node_differential.cpp
	source/core/node_differentiation.cpp
	source/core/node_list.cpp
	source/core/node_manager.cpp
	source/core/node_reference.cpp
	source/core/node_value.cpp
	source/core/operation.cpp
	source/core/operation_holder.cpp
	source/core/rvalue.cpp
	# source/core/shader.cpp
	source/core/special_tokens.cpp
	source/core/types.cpp
	source/core/wildcard.cpp
	source/core/variable_cluster.cpp

	source/lang/cc_keywords.cpp
	source/lang/cc_parser.cpp
	source/lang/compilation.cpp
	source/lang/default_feeders.cpp
	source/lang/error_handling.cpp
	source/lang/feeder.cpp
	source/lang/helpers.cpp
	source/lang/keywords.cpp
	source/lang/mld_parser.cpp
	source/lang/parser.cpp
	source/lang/source.cpp
	source/lang/vm.cpp

	source/std/functions.cpp
	source/std/interval.cpp
	source/std/linalg.cpp

	interpreter/builtin/basic_io.cpp
	
	# glad/glad.c
)

# Compile shared and static library
add_library(zhp-shared SHARED ${ZHETAPI_SOURCE})

target_link_libraries(zhp-shared ${PNG_LIBRARY})
target_link_libraries(zhp-shared ${CMAKE_DL_LIBS})
target_link_libraries(zhp-shared sfml-graphics sfml-audio)

SET_TARGET_PROPERTIES(zhp-shared PROPERTIES
   OUTPUT_NAME zhp CLEAN_DIRECT_OUTPUT 1)

add_library(zhp-static STATIC ${ZHETAPI_SOURCE})

SET_TARGET_PROPERTIES(zhp-static PROPERTIES
   OUTPUT_NAME zhp CLEAN_DIRECT_OUTPUT 1)
