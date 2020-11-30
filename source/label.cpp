#include <hidden/label.hpp>

namespace zhetapi {

	::std::string strlabs[] = {
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
		
		// References
		"variable reference",
		"node reference",

		// Miscellaneious
		"constant",
		"function",
		"operation constant",

		// Operations
		"dot",
		"separable",
		"multiplied",
		"divided",
		"power",
		"trigonometric",
		"hyperbolic",
		"natural logarithm",
		"binary logarithm",
		"constant base logarithm",
		"variable base logarithm",

		// Variable
		"variable",
		"variable constant",
	};

	bool is_constant(lbl type)
	{
		if ((type >= l_constant_integer)
				&& (type <= l_constant_matrix_complex_real)
				|| (type == l_operation_constant)
				|| (type == l_variable_constant))
			return true;

		return false;
	}

	bool is_constant_operand(lbl type)
	{
		if ((type >= l_constant_integer)
				&& (type <= l_constant_matrix_complex_real))
			return true;

		return false;
	}

	bool is_variable(lbl type)
	{
		return !is_constant(type);
	}

}