#ifndef LABEL_H_
#define LABEL_H_

// C/C++ headers
#include <string>

// Engine headers
#include <token.hpp>

#include <hidden/types.hpp>

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

		// References
		l_variable_reference,
		l_node_reference,

		// Miscellaneious
		l_constant,
		l_function,
		l_operation_constant,

		// Operations
		l_dot,
		l_separable,
		l_multiplied,
		l_divided,
		l_power,
		l_trigonometric,
		l_natural_log,
		l_binary_log,
		l_constant_base_log,
		l_variable_base_log,

		// Variables
		l_variable,		// Regular variable
		l_variable_constant,	// Variable as a constant (differentiation)
	};

	/**
	 * @brief String representations
	 * of the corresponding labels
	 * in the label enumeration.
	 */
	extern ::std::string strlabs[];

	// Forward declarations
	bool is_constant(lbl);
	bool is_constant_operand(lbl);
	bool is_variable(lbl);

	// Template functions
	template <class T, class U>
	lbl constant_label(Token *tptr)
	{
		// Complex numbers
		if (dynamic_cast <typename types <T, U> ::opd_z *> (tptr))
			return l_constant_integer;
		if (dynamic_cast <typename types <T, U> ::opd_q *> (tptr))
			return l_constant_rational;
		if (dynamic_cast <typename types <T, U> ::opd_r *> (tptr))
			return l_constant_real;
		if (dynamic_cast <typename types <T, U> ::opd_cz *> (tptr))
			return l_constant_complex_integer;
		if (dynamic_cast <typename types <T, U> ::opd_cq *> (tptr))
			return l_constant_complex_rational;
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

	template <class T, class U>
	Token *true_token(Token *tptr)
	{
		typename types <T, U> ::opd_z *oz = dynamic_cast <typename types <T, U> ::opd_z *> (tptr);
		if (oz != nullptr)
			return oz;

		return nullptr;
	}

}

#endif
