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

// Import the lex classes
#include "tokens_nota.hpp"

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
