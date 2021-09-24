#include "../../engine/lang/ltag.hpp"

namespace zhetapi {

std::string strlex[] = {
	"DONE", "NEWLINE", "COMMA",

	// TODO: order operations in terms of precedence
	"PLUS", "MINUS", "TIMES", "DIVIDE",

	// op = operations
	"PLUS_EQ", "MINUS_EQ", "TIMES_EQ", "DIVIDE_EQ",
	"LOGIC AND", "BIT AND",
	"LOGIC OR", "BIT OR", "LOGIC EQ",
	"ASSIGN EQ", "LOGIC NOT",
	"BIT NOT", "GEQ", "GE", "LEQ", "LE",
	"LPAREN", "RPAREN",
	"LBRACE", "RBRACE",

	"IDENTIFIER", "PRIMITIVE",
	"ALGORITHM"
};

// Functions
void free_ltag(void *data)
{
	LexTag ltag = get_ltag(data);
	if (ltag == IDENTIFIER)
		delete (IdentifierTag *) data;
	else if (ltag == PRIMITIVE)
		delete (PrimitiveTag *) data;
	else
		delete (NormalTag *) data;
}

std::string to_string(void *data)
{
	LexTag ltag = get_ltag(data);
	switch (ltag) {
	case PRIMITIVE:
		return "PRIMITIVE (" + PrimitiveTag::cast(data).str() + ")";
	case IDENTIFIER:
		return "IDENTIFIER (" + IdentifierTag::cast(data) + ")";
	default:
		break;
	}

	return strlex[ltag];
}

}
