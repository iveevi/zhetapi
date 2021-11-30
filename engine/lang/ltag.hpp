#ifndef LTAG_H_
#define LTAG_H_

// Standard headers
#include <string>

// Engine headers
#include "../core/primitive.hpp"
#include "../core/object.hpp"
#include "../core/variant.hpp"

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

	IDENTIFIER,
	ALGORITHM,

	// Built-in operands
	PRIMITIVE,
	STRING,
	OBJECT,		// Non-built-in type

	// Grammatical structures
	gr_start,
	gr_statements,
	gr_statement,
	gr_assignment,
	gr_expression,
	gr_simple_expression,
	gr_term,
	gr_factor,
	gr_full_factor,
	gr_closed_factor,
	gr_operand
};

// String codes for enumerations
extern std::string strlex[];

// Helper functions
inline constexpr LexTag get_ltag(void *ltag)
{
	return *((LexTag *) ltag);
}

inline std::string str_ltag(void *ltag)
{
	return strlex[get_ltag(ltag)];
}

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

struct PrimitiveTag {
	size_t id = PRIMITIVE;
	Primitive value;

	PrimitiveTag(Primitive x) : value(x) {}

	// Value function
	static Primitive cast(void *ptr) {
		return ((PrimitiveTag *) ptr)->value;
	}
};

struct ObjectTag {
	size_t id;
	Object value;

	ObjectTag(Object x, size_t type = OBJECT)
		: value(x), id(type) {}

	// Value function
	static Object cast(void *ptr) {
		return ((ObjectTag *) ptr)->value;
	}
};

// Casting variant tags
constexpr Variant vt_cast(void *ptr)
{
	switch (get_ltag(ptr)) {
	case PRIMITIVE:
		return &((PrimitiveTag *) ptr)->value;
	case OBJECT:
		return &((ObjectTag *) ptr)->value;
	default:
		break;
	}

	return nullptr;
}

// Forward declarations
void free_ltag(void *);
std::string to_string(void *);

}

#endif
