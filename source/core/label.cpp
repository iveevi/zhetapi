#include <core/label.hpp>
#include <core/node.hpp>
#include <core/operation_holder.hpp>

namespace zhetapi {

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

	"constant boolean",
	"constant string",
	
	// References
	"variable reference",
	"node reference",

	// Algorithm
	"sequential",
	"assignment chain",
	"lvalue",
	"if",
	"else",
	"elif",
	"branch",

	// Miscellaneious
	"constant",
	"function",
	"registrable",
	"operation constant",
	"wildcard",

	// Operations
	"dot",
	"separable",
	"multiplied",
	"divided",
	"power",
	"miscellaneous power",
	"trigonometric",
	"hyperbolic",
	"natural logarithm",
	"binary logarithm",
	"constant base logarithm",
	"variable base logarithm",

	// Variable
	"variable",
	"variable constant",

	// Modifiers
	"post modifier",
	"pre modifier",

	// Calculus
	"differential"
};

bool is_constant(lbl type)
{
	return ((type >= l_constant_integer)
			&& (type <= l_constant_matrix_complex_real))
			|| (type == l_operation_constant)
			|| (type == l_variable_constant);
}

bool is_constant_operand(lbl type)
{
	return (type >= l_constant_integer)
			&& (type <= l_constant_matrix_complex_real);
}

bool is_variable(lbl type)
{
	return !is_constant(type);
}

bool is_commutative(const node &tree)
{
	operation_holder *ophptr = tree.cast <operation_holder> ();

	if (!ophptr)
		return false;
	
	return (ophptr->code == mul || ophptr->code == add);
}

lbl constant_label(Token *tptr)
{
	// Complex numbers
	if (dynamic_cast <opd_z *> (tptr))
		return l_constant_integer;
	if (dynamic_cast <opd_q *> (tptr))
		return l_constant_rational;
	if (dynamic_cast <opd_r *> (tptr))
		return l_constant_real;
	if (dynamic_cast <opd_cz *> (tptr))
		return l_constant_complex_integer;
	if (dynamic_cast <opd_cq *> (tptr))
		return l_constant_complex_rational;
	if (dynamic_cast <opd_cr *> (tptr))
		return l_constant_complex_real;
	
	// Vectors
	if (dynamic_cast <opd_v_z *> (tptr))
		return l_constant_vector_integer;
	if (dynamic_cast <opd_v_q *> (tptr))
		return l_constant_vector_rational;
	if (dynamic_cast <opd_v_r *> (tptr))
		return l_constant_vector_real;
	if (dynamic_cast <opd_v_cz *> (tptr))
		return l_constant_vector_complex_integer;
	if (dynamic_cast <opd_v_cq *> (tptr))
		return l_constant_vector_complex_rational;
	if (dynamic_cast <opd_v_cr *> (tptr))
		return l_constant_vector_complex_real;
	
	// Matrices
	if (dynamic_cast <opd_m_z *> (tptr))
		return l_constant_matrix_integer;
	if (dynamic_cast <opd_m_q *> (tptr))
		return l_constant_matrix_rational;
	if (dynamic_cast <opd_m_r *> (tptr))
		return l_constant_matrix_real;
	if (dynamic_cast <opd_m_cz *> (tptr))
		return l_constant_matrix_complex_integer;
	if (dynamic_cast <opd_m_cq *> (tptr))
		return l_constant_matrix_complex_rational;
	if (dynamic_cast <opd_m_cr *> (tptr))
		return l_constant_matrix_complex_real;
	
	// Other
	if (dynamic_cast <opd_b *> (tptr))
		return l_constant_boolean;
	if (dynamic_cast <opd_s *> (tptr))
		return l_constant_string;

	return l_none;
}

}
