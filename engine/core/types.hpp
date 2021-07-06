#ifndef TYPES_H_
#define TYPES_H_

// C/C++ headers
#include <functional>
#include <typeindex>

// Engine headers
#include "../operand.hpp"
#include "raw_types.hpp"

namespace zhetapi {

// TODO: change to namespace
struct types {
	static Token *one();
	
	static bool is_zero(Token *);
	static bool is_one(Token *);

	static const char *symbol(const std::type_index &);
};

std::string type_name(const std::type_index &);

// Forward declarations
class Function;
class algorithm;
class Registrable;
class Module;
class Collection;

// All valid tokens
// IDs are increasing in this order
using OpZ = Operand <Z>;
using OpQ = Operand <Q>;
using OpR = Operand <R>;

using OpB = Operand <B>;
using OpS = Operand <S>;

using OpCmpZ = Operand <CmpZ>;
using OpCmpQ = Operand <CmpQ>;
using OpCmpR = Operand <CmpR>;

using OpVecZ = Operand <VecZ>;
using OpVecQ = Operand <VecQ>;
using OpVecR = Operand <VecR>;
using OpVecCmpZ = Operand <VecCmpZ>;
using OpVecCmpQ = Operand <VecCmpQ>;
using OpVecCmpR = Operand <VecCmpR>;

using OpMatZ = Operand <MatZ>;
using OpMatQ = Operand <MatQ>;
using OpMatR = Operand <MatR>;
using OpMatCmpZ = Operand <MatCmpZ>;
using OpMatCmpQ = Operand <MatCmpQ>;
using OpMatCmpR = Operand <MatCmpR>;

// Comparison (for containers)
class TokenEqual {
public:
	bool operator()(Token *a, Token *b) const {
		return tokcmp(a, b);
	}
};

// Hashing (for containers)
class TokenHash {
public:
	size_t operator()(Token *a) const {
		// TODO: add a macro for all of this
		OpZ *opz;
		if ((opz = dynamic_cast <OpZ *> (a)))
			return std::hash <Z> {} (opz->get());

		OpR *opr;
		if ((opr = dynamic_cast <OpR *> (a)))
			return std::hash <R> {} (opr->get());

		OpS *ops;
		if ((ops = dynamic_cast <OpS *> (a)))
			return std::hash <S> {} (ops->get());

		throw unhashable(typeid(*a));
	}

	// Exceptions
	class unhashable : public std::runtime_error {
	public:
		unhashable(const std::type_index &ti)
				: std::runtime_error("Cannot hash type <"
						+ std::string(types::symbol(ti))
						+ ">") {}
	};
};

using Targs = std::vector <Token *>;
using TTargs = std::vector <std::pair <Token *, Token *>>;
using THTable = std::unordered_map <Token *, Token *, TokenHash, TokenEqual>;

#define	ONE	1

}

#endif
