#include "../../engine/lang/ltag.hpp"

namespace zhetapi {

std::string strlex[] = {
	"DONE",

	// TODO: order operations in terms of precedence
	"PLUS", "MINUS", "TIMES", "DIVIDE",
	"FACTORIAL",

	// Dual character binary operations
	"LOGIC_AND", "LOGIC_OR",
	"LOGIC_EQ",

	// Operation equals
	"PLUS_EQ", "MINUS_EQ", "TIMES_EQ", "DIVIDE_EQ",

	// Bitwise operations
	"BITWISE_AND", "BITWISE_OR", "BITWISE_XOR", "BITWISE_NOT",

	// Comparison operations
	"EQ", "NEQ", "GT", "LT", "GTE", "LTE",

	// Kewords
	"ALGORITHM",

	// Miscelaneous
	"LPAREN", "RPAREN",
	"LBRACE", "RBRACE",
	"LBRACKET", "RBRACKET",

	"NEWLINE",
	"COMMA",
	"ASSIGNMENT",
	"IDENTIFIER",

	// Built-in operands
	"PRIMITIVE",
	"STRING",
	"OBJECT",

	// Grammatical structures
	"GR-START",
	"GR-STATEMENTS",
	"GR-STATEMENT",
	"GR-ASSIGNMENT",
	"GR-EXPRESSION",
	"GR-SIMPLE-EXPRESSION",
	"GR-TERM",
	"GR-FACTOR",
	"GR-FULL-FACTOR",
	"GR-CLOSED-FACTOR",
	"GR-OPERAND"
};

// Functions
void free_ltag(void *data)
{
	LexTag ltag = get_ltag(data);
	if (ltag == IDENTIFIER::id)
		delete (IdentifierTag *) data;
	else if (ltag == PRIMITIVE::id)
		delete (PrimitiveTag *) data;
	else
		delete (NormalTag *) data;
}

std::string to_string(void *data)
{
	LexTag ltag = get_ltag(data);
	switch (ltag) {
	case PRIMITIVE::id:
		return "PRIMITIVE (" + PrimitiveTag::cast(data).str() + ")";
	case IDENTIFIER::id:
		return "IDENTIFIER (" + IdentifierTag::cast(data) + ")";
	default:
		break;
	}

	return strlex[ltag];
}

}
