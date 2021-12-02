#ifndef LTAG_H_
#define LTAG_H_

// Standard headers
#include <string>
#include <sys/types.h>

// Engine headers
#include "../core/primitive.hpp"
#include "../core/object.hpp"
#include "../core/variant.hpp"


namespace zhetapi {

// Lex tag identifier
using LexTag = uint64_t;

// Lex class structure
template <LexTag code>
struct LexClass {
	static constexpr LexTag id = code;
};

// Import the lex classes
#include "tokens_nota.hpp"

// TODO: a macro to create a counter
constexpr fameta::counter <__COUNTER__, 0> C;

#define MK_TAG(tag) struct tag : public LexClass <C.next <__COUNTER__> ()> {};
#define MK_TAGS(...) MK_TAG(_1)

struct DONE : public LexClass <0> {};
struct PLUS : public LexClass <1> {};
struct MINUS : public LexClass <2> {};
struct TIMES : public LexClass <3> {};
struct DIVIDE : public LexClass <4> {};
struct FACTORIAL : public LexClass <5> {};

struct LOGIC_AND : public LexClass <6> {};
struct LOGIC_OR : public LexClass <7> {};
struct LOGIC_EQ : public LexClass <8> {};

struct PLUS_EQ : public LexClass <9> {};
struct MINUS_EQ : public LexClass <10> {};
struct TIMES_EQ : public LexClass <11> {};
struct DIVIDE_EQ : public LexClass <12> {};

struct BIT_AND : public LexClass <13> {};
struct BIT_OR : public LexClass <14> {};
struct BT_XOR : public LexClass <15> {};
struct BIT_NOT : public LexClass <16> {};

struct EQ : public LexClass <17> {};
struct NEQ : public LexClass <18> {};
struct GT : public LexClass <19> {};
struct LT : public LexClass <20> {};
struct GTE : public LexClass <21> {};
struct LTE : public LexClass <22> {};

struct ALG : public LexClass <23> {};

struct LPAREN : public LexClass <24> {};
struct RPAREN : public LexClass <25> {};
struct LBRACE : public LexClass <26> {};
struct RBRACE : public LexClass <27> {};
struct LBRACKET : public LexClass <28> {};
struct RBRACKET : public LexClass <29> {};

struct NEWLINE : public LexClass <30> {};
struct COMMA : public LexClass <31> {};
struct ASSIGN_EQ : public LexClass <32> {};
struct IDENTIFIER : public LexClass <33> {};

struct PRIMITIVE : public LexClass <34> {};
struct STRING : public LexClass <35> {};
struct OBJECT : public LexClass <36> {};

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
	size_t id = IDENTIFIER::id;
	std::string ident;

	IdentifierTag(const std::string str) : ident(str) {}

	// Value function
	static std::string cast(void *ptr) {
		return ((IdentifierTag *) ptr)->ident;
	}
};

struct PrimitiveTag {
	size_t id = PRIMITIVE::id;
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

	ObjectTag(Object x, size_t type = OBJECT::id)
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
	case PRIMITIVE::id:
		return &((PrimitiveTag *) ptr)->value;
	case OBJECT::id:
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
