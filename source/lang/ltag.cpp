#include "../../engine/lang/ltag.hpp"

namespace zhetapi {

std::string strlex[] = {
	"DONE", "COMMA",
	"PLUS", "MINUS",
	"PLUS_EQ", "MINUS_EQ",
	"LOGIC AND", "BIT AND",
	"LOGIC OR", "BIT OR", "LOGIC EQ",
	"ASSIGN EQ", "LOGIC NOT",
	"BIT NOT", "GEQ", "GE", "LEQ", "LE",
	"LPAREN", "RPAREN",
	"LBRACE", "RBRACE",
	"IDENTIFIER", "INTEGER",
	"ALGORITHM"
};

// Functions
void free_ltag(void *ltag)
{
	LexTag id = *((LexTag *) ltag);
	if (id == IDENTIFIER)
		delete (Identifier *) ltag;
	else if (id == INTEGER)
		delete (Integer *) ltag;
	else
		delete (Normal *) ltag;	
}

}