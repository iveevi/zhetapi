#ifndef LABEL_H_
#define LABEL_H_

#include <types.hpp>

namespace zhetapi {

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

		"constant complex integer",
		"constant complex rational",
		"constant complex real",

		"constant vector integer",
		"constant vector rational",
		"constant vector real",

		"constant vector complex integer",
		"constant vector complex rational",
		"constant vector complex real",

		"constant matrix integer",
		"constant matrix rational",
		"constant matrix real",

		"constant matrix complex integer",
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

	bool is_constant(lbl type)
	{
		if ((type >= l_constant_integer)
				&& (type <= l_constant_matrix_complex_real)
				|| (type == l_operation_constant))
			return true;

		return false;
	}

	bool is_variable(lbl type)
	{
		return !is_constant(type);
	}

	template <class T, class U>
	lbl constant_label(token *tptr)
	{
		using namespace std;

		cout << "Labeling: " << tptr << ", also equals " << tptr->str() << endl;

		// Complex numbers
		if (dynamic_cast <typename types <T, U> ::opd_z *> (tptr))
			return l_constant_integer;
		if (dynamic_cast <typename types <T, U> ::opd_q *> (tptr))
			return l_constant_rational;
		if (dynamic_cast <typename types <T, U> ::opd_r *> (tptr))
			return l_constant_real;

		if (dynamic_cast <typename types <T, U> ::opd_cz *> (tptr)) {
			cout << "\tHERE!! - integer" << endl;
			return l_constant_complex_integer;
		}

		if (dynamic_cast <typename types <T, U> ::opd_cq *> (tptr)) {
			cout << "\tHERE!! - rational" << endl;
			return l_constant_complex_rational;
		}

		if (dynamic_cast <typename types <T, U> ::opd_cr *> (tptr))
			return l_constant_complex_real;
		
		// Vectors
		if (dynamic_cast <typename types <T, U> ::opd_v_z *> (tptr))
			return l_constant_vector_integer;
		if (dynamic_cast <typename types <T, U> ::opd_v_q *> (tptr))
			return l_constant_vector_rational;
		if (dynamic_cast <typename types <T, U> ::opd_v_r *> (tptr))
			return l_constant_vector_real;
		if (dynamic_cast <typename types <T, U> ::opd_v_cz *> (tptr))
			return l_constant_vector_complex_integer;
		if (dynamic_cast <typename types <T, U> ::opd_v_cq *> (tptr))
			return l_constant_vector_complex_rational;
		if (dynamic_cast <typename types <T, U> ::opd_v_cr *> (tptr))
			return l_constant_vector_complex_real;
		
		// Matrices
		if (dynamic_cast <typename types <T, U> ::opd_m_z *> (tptr))
			return l_constant_matrix_integer;
		if (dynamic_cast <typename types <T, U> ::opd_m_q *> (tptr))
			return l_constant_matrix_rational;
		if (dynamic_cast <typename types <T, U> ::opd_m_r *> (tptr))
			return l_constant_matrix_real;
		if (dynamic_cast <typename types <T, U> ::opd_m_cz *> (tptr))
			return l_constant_matrix_complex_integer;
		if (dynamic_cast <typename types <T, U> ::opd_m_cq *> (tptr))
			return l_constant_matrix_complex_rational;
		if (dynamic_cast <typename types <T, U> ::opd_m_cr *> (tptr))
			return l_constant_matrix_complex_real;

		return l_none;
	}

}

#endif
