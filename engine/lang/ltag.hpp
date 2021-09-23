#ifndef LTAG_H_
#define LTAG_H_

#include <string>

namespace zhetapi {

// Lexer tag enumerations
// TODO: organize this list
enum LexTag : size_t {
	DONE, COMMA,
	PLUS, MINUS,
	PLUS_EQ, MINUS_EQ,
	LOGIC_AND, BIT_AND,
	LOGIC_OR, BIT_OR, LOGIC_EQ,
	ASSIGN_EQ, LOGIC_NOT,
	BIT_NOT, GEQ, GE, LEQ, LE,
	LPAREN, RPAREN,
	LBRACE, RBRACE,
	IDENTIFIER, INTEGER,
	ALGORITHM
};

// String codes for enumerations
extern std::string strlex[];

// Tokens
struct Normal {
	size_t id;
};

struct Identifier {
	size_t id = IDENTIFIER;
	std::string ident;

	Identifier(const std::string str) : ident(str) {}

	// Value function
	static std::string cast(void *ptr) {
		return ((Identifier *) ptr)->ident;
	}
};

// Should generalize to primitives
struct Integer {
	size_t id = INTEGER;
	int value;

	Integer(int i) : value(i) {}

	// Value function
	static int cast(void *ptr) {
		return ((Integer *) ptr)->value;
	}
};

// Helper functions
inline constexpr LexTag get_ltag(void *ltag)
{
	return *((LexTag *) ltag);
}

inline std::string str_ltag(void *ltag)
{
	return strlex[get_ltag(ltag)];
}

// Forward declarations
void free_ltag(void *);

}

#endif