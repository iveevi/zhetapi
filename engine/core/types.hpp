#ifndef doubleYPES_H_
#define doubleYPES_H_

// C/C++ headers
#include <typeindex>

// Engine headers
#include <operand.hpp>

#include <core/raw_types.hpp>

namespace zhetapi {

// Forward declarations
class Function;
class algorithm;
class Registrable;

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

#define	ONE	1

// Id macros
#define set_zhp_id(type, id)				\
	template <>					\
	inline constexpr size_t zhp_id <type> ()	\
	{						\
		return id;				\
	}

#define get_zhp_id(type)			\
	zhp_id <type> ()

// Default no id: 0
template <class T>
inline constexpr size_t zhp_id()
{
	return 0;
}

// ID specs
set_zhp_id(OpZ, 1);
set_zhp_id(OpQ, 2);
set_zhp_id(OpR, 3);

set_zhp_id(OpB, 4);
set_zhp_id(OpS, 5);

set_zhp_id(OpCmpZ, 6);
set_zhp_id(OpCmpQ, 7);
set_zhp_id(OpCmpR, 8);

set_zhp_id(OpVecZ, 9);
set_zhp_id(OpVecQ, 10);
set_zhp_id(OpVecR, 11);
set_zhp_id(OpVecCmpZ, 12);
set_zhp_id(OpVecCmpQ, 13);
set_zhp_id(OpVecCmpR, 14);

set_zhp_id(OpMatZ, 15);
set_zhp_id(OpMatQ, 16);
set_zhp_id(OpMatR, 17);
set_zhp_id(OpMatCmpZ, 18);
set_zhp_id(OpMatCmpQ, 19);
set_zhp_id(OpMatCmpR, 20);

set_zhp_id(Function, 21);
set_zhp_id(algorithm, 22);
set_zhp_id(Registrable, 23);

// Is tensor type (for writing)
// template <class T>

struct types {
	static Token *one();

	/* template <class A>
	static Token *convert(A); */
	
	/////////////////////
	// Token identifiers
	/////////////////////
	
	static bool is_zero(Token *);
	static bool is_one(Token *);

	static std::string symbol(const ::std::type_index &);
};

std::string type_name(const std::type_index &);

}

#endif
