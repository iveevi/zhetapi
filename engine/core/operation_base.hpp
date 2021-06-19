#ifndef OPERATION_BASE_H_
#define OPERATION_BASE_H_

// C/C++ headers
#include <cstdint>
#include <stdexcept>
#include <unordered_map>

// Engine headers
#include "../token.hpp"
#include "types.hpp"

namespace zhetapi {

using OvID = uint16_t;

// ID function (run-time)
inline constexpr OvID overload_hash(Token *arg1, Token *arg2)
{
	// Args should always have 4 elements
	return (arg1 ? arg1->id() : 0)
		+ ((arg2 ? arg2->id() : 0) << 8);
}

// ID function (compile-time)
template <class O1 = int, class O2 = int>
inline constexpr uint32_t overload_hash()
{
	return (zhp_id <O1> ())
		+ (zhp_id <O2> () << 8);
}

// No capture lambdas
using Operation = Token *(*)(Token *, Token *);

// Up to 256 operations (each with 4 operations)
enum OpCode : uint8_t {
	l_add,
	l_sub,
	l_mul,
	l_div
};

// Operation to string (add formatting of arguments for display)
extern const char *OpStr[];

using OverloadBase = const std::map <OvID, Operation>;

// Up to 256 operations
extern const OverloadBase OperationBase[];

// Type symbol (TODO: add constexpr)
inline const char *token_symbol(const Token *const tptr)
{
	// Complex numbers
	// TODO: use ID and a hash table
	if (tptr->id() == zhp_id <OpZ> ())
		return "Z";
	if (tptr->id() == zhp_id <OpQ> ())
		return "Q";
	if (tptr->id() == zhp_id <OpR> ())
		return "R";
	if (tptr->id() == zhp_id <OpCmpZ> ())
		return "CZ";
	if (tptr->id() == zhp_id <OpCmpQ> ())
		return "CQ";
	if (tptr->id() == zhp_id <OpCmpR> ())
		return "CR";
	
	// Vectors
	if (tptr->id() == zhp_id <OpVecZ> ())
		return "VZ";
	if (tptr->id() == zhp_id <OpVecQ> ())
		return "VQ";
	if (tptr->id() == zhp_id <OpVecR> ())
		return "VR";
	if (tptr->id() == zhp_id <OpVecCmpZ> ())
		return "VCZ";
	if (tptr->id() == zhp_id <OpVecCmpQ> ())
		return "VCQ";
	if (tptr->id() == zhp_id <OpVecCmpR> ())
		return "VCR";
	
	// Matrices
	if (tptr->id() == zhp_id <OpMatZ> ())
		return "MZ";
	if (tptr->id() == zhp_id <OpMatQ> ())
		return "MQ";
	if (tptr->id() == zhp_id <OpMatR> ())
		return "MR";
	if (tptr->id() == zhp_id <OpMatCmpZ> ())
		return "MCZ";
	if (tptr->id() == zhp_id <OpMatCmpQ> ())
		return "MCQ";
	if (tptr->id() == zhp_id <OpMatCmpR> ())
		return "MCR";

	return "?";
}

// Signature generation
inline std::string gen_overload_msg(OpCode code, Token *arg1, Token *arg2)
{
	std::string msg = "Unknown overload (";

	msg += std::string(arg1 ? token_symbol(arg1) : "Null") + ", ";
	msg += std::string(arg2 ? token_symbol(arg2) : "Null") + ", ";

	return msg + ") for operation " + OpStr[code] + ".";
}

// Value function
inline Token *do_optn(OpCode code, Token *arg1, Token *arg2)
{
	OvID id = overload_hash(arg1, arg2);
	
	const OverloadBase ob = OperationBase[code];
	if (ob.find(id) == ob.end())
		throw std::runtime_error(gen_overload_msg(code, arg1, arg2));

	return (ob.at(id))(arg1, arg2);
}

}

#endif
