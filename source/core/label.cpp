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
	"rvalue",
	"if",
	"else",
	"elif",
	"branch",
	"while loop",
	"for loop",
	"break loop",
	"continue loop",
	"return",

	// "in" operator
	"generator-in",

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
	// TODO: Use static hash table later or smthing

	// Complex numbers
	if (dynamic_cast <OpZ *> (tptr))
		return l_constant_integer;
	if (dynamic_cast <OpQ *> (tptr))
		return l_constant_rational;
	if (dynamic_cast <OpR *> (tptr))
		return l_constant_real;
	if (dynamic_cast <OpCmpZ *> (tptr))
		return l_constant_complex_integer;
	if (dynamic_cast <OpCmpQ *> (tptr))
		return l_constant_complex_rational;
	if (dynamic_cast <OpCmpR *> (tptr))
		return l_constant_complex_real;
	
	// Vectors
	if (dynamic_cast <OpVecZ *> (tptr))
		return l_constant_vector_integer;
	if (dynamic_cast <OpVecQ *> (tptr))
		return l_constant_vector_rational;
	if (dynamic_cast <OpVecR *> (tptr))
		return l_constant_vector_real;
	if (dynamic_cast <OpVecCmpZ *> (tptr))
		return l_constant_vector_complex_integer;
	if (dynamic_cast <OpVecCmpQ *> (tptr))
		return l_constant_vector_complex_rational;
	if (dynamic_cast <OpVecCmpR *> (tptr))
		return l_constant_vector_complex_real;
	
	// Matrices
	if (dynamic_cast <OpMatZ *> (tptr))
		return l_constant_matrix_integer;
	if (dynamic_cast <OpMatQ *> (tptr))
		return l_constant_matrix_rational;
	if (dynamic_cast <OpMatR *> (tptr))
		return l_constant_matrix_real;
	if (dynamic_cast <OpMatCmpZ *> (tptr))
		return l_constant_matrix_complex_integer;
	if (dynamic_cast <OpMatCmpQ *> (tptr))
		return l_constant_matrix_complex_rational;
	if (dynamic_cast <OpMatCmpR *> (tptr))
		return l_constant_matrix_complex_real;
	
	// Other
	if (dynamic_cast <OpB *> (tptr))
		return l_constant_boolean;
	if (dynamic_cast <OpS *> (tptr))
		return l_constant_string;

	return l_none;
}

}
