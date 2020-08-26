#ifndef LABEL_H_
#define LABEL_H_

/**
 * @brief The enumeration
 * label is used to label nodes
 * of an expression tree.
 *
 * Clarifications:
 *  - l_logarithmic implies a
 *  logarithm of a Variable base.
 *  - l_constant_logarithmic implies
 *  a logarithm of a constant base.
 */
enum nd_label {
	// Default
	l_none,

	// Constants
	l_constant_integer,
	l_constant_rational,
	l_constant_real,
	l_constant_complex_rational,
	l_constant_complex_real,
	l_constant_vector_rational,
	l_constant_vector_real,
	l_constant_vector_complex_rational,
	l_constant_vector_complex_real,
	l_constant_matrix_rational,
	l_constant_matrix_real,
	l_constant_matrix_complex_rational,
	l_constant_matrix_complex_real,

	// Operations
	l_dot,

	// off

	l_matrix_uncoded,	// leave martix as nodes (with Variables)
	l_vector_uncoded,	// leave vector as nodes, then decode once substituion is performed

	l_constant,		// keep to prevent errors
	l_power,
	l_divided,
	l_variable,
	l_function,
	l_exp,
	l_polynomial,
	l_separable,
	l_multiplied,
	l_exponential,
	l_logarithmic,
	l_trigonometric,
	l_power_uncertain,
	l_function_constant,
	l_operation_constant,
	l_constant_logarithmic,
	l_summation,
	l_summation_Variable,
	l_summation_function,
	l_factorial
};

/**
 * @brief String representations
 * of the corresponding labels
 * in the label enumeration.
 */
std::string strlabs[] = {
	// Default
	"none",

	// Constants
	"constant integer",
	"constant rational",
	"constant real",
	"constant complex rational",
	"constant complex real",
	"constant vector rational",
	"constant vector real",
	"constant vector complex rational",
	"constant vector complex real",
	"constant matrix rational",
	"constant matrix real",
	"constant matrix complex rational",
	"constant matrix complex real",

	// Operations
	"dot",

	// off
	"vector uncoded",
	"matrix uncoded",

	"constant",
	
	"power",
	"divided",
	"variable",
	"function",
	"exponent",
	"polynomic",
	"separable",
	"multiplied",
	"exponential",
	"logarithmic",
	"trigonometric",
	"power uncertain",
	"function constant",
	"operation constant",
	"constant logarithmic",
	"summation",
	"summation variable",
	"summation function",
	"factorial"
};

bool is_constant(nd_label type)
{
	if ((type >= l_constant_integer) &&
			(type <= l_constant_matrix_complex_real))
		return true;

	return false;
}

bool is_variable(nd_label type)
{
	return !is_constant(type);
}

#endif
