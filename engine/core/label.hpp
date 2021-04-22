#ifndef LABEL_H_
#define LABEL_H_

// C/C++ headers
#include <string>

// Engine headers
#include <token.hpp>

#include <core/types.hpp>

namespace zhetapi {

class node;

enum lbl {
	// Default
	l_none,

	// Constants
	l_constant_integer,
	l_constant_rational,
	l_constant_real,

	l_constant_complex_integer,
	l_constant_complex_rational,
	l_constant_complex_real,

	l_constant_vector_integer,
	l_constant_vector_rational,
	l_constant_vector_real,

	l_constant_vector_complex_integer,
	l_constant_vector_complex_rational,
	l_constant_vector_complex_real,

	l_constant_matrix_integer,
	l_constant_matrix_rational,
	l_constant_matrix_real,

	l_constant_matrix_complex_integer,
	l_constant_matrix_complex_rational,
	l_constant_matrix_complex_real,

	l_constant_boolean,
	l_constant_string,

	// References
	l_variable_reference,
	l_node_reference,

	// Algorithm
	l_sequential,				// Sequential node
	l_assignment_chain,			// Assignment statements
	l_lvalue,
	l_if_branch,
	l_else_branch,
	l_elif_branch,
	l_branch,

	// Miscellaneious
	l_constant,
	l_function,
	l_registrable,
	l_operation_constant,
	l_wildcard,

	// Operations
	l_dot,
	l_separable,
	l_multiplied,
	l_divided,
	l_power,
	l_power_misc,
	l_trigonometric,
	l_hyperbolic,
	l_natural_log,
	l_binary_log,
	l_constant_base_log,
	l_variable_base_log,

	// Variables
	l_variable,		// Regular variable
	l_variable_constant,	// Variable as a constant (differentiation)

	// Modifier,
	l_post_modifier,
	l_pre_modifier,

	// Calculus
	l_differential
};

/**
 * @brief String representations
 * of the corresponding labels
 * in the label enumeration.
 */
extern std::string strlabs[];

// Forward declarations
bool is_constant(lbl);
bool is_constant_operand(lbl);
bool is_variable(lbl);

bool is_commutative(const node &);

lbl constant_label(Token *);

}

#endif
