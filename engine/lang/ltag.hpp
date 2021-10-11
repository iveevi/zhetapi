#ifndef LTAG_H_
#define LTAG_H_

// Standard headers
#include <string>

// Engine headers
#include "../core/primitive.hpp"

namespace zhetapi {

// Lexer tag enumerations
// TODO: organize this list
enum LexTag : size_t {
	DONE, NEWLINE, COMMA,

	// regular binary operations
	PLUS, MINUS, TIMES, DIVIDE,

	// op = operations
	PLUS_EQ, MINUS_EQ, TIMES_EQ, DIVIDE_EQ,
	LOGIC_AND, BIT_AND,
	LOGIC_OR, BIT_OR, LOGIC_EQ,
	ASSIGN_EQ, LOGIC_NOT,
	BIT_NOT, GEQ, GE, LEQ, LE,
	LPAREN, RPAREN,
	LBRACE, RBRACE,
	ARG_PACK,

	IDENTIFIER, PRIMITIVE,
	ALGORITHM
};

// String codes for enumerations
extern std::string strlex[];

// Tokens
struct NormalTag {
	size_t id;
};

struct IdentifierTag {
	size_t id = IDENTIFIER;
	std::string ident;

	IdentifierTag(const std::string str) : ident(str) {}

	// Value function
	static std::string cast(void *ptr) {
		return ((IdentifierTag *) ptr)->ident;
	}
};

// Should generalize to primitives
struct PrimitiveTag {
	size_t id = PRIMITIVE;
	Primitive value;

	PrimitiveTag(Primitive x) : value(x) {}

	// Value function
	static Primitive cast(void *ptr) {
		return ((PrimitiveTag *) ptr)->value;
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
std::string to_string(void *);

}

#endif
