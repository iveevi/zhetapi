#ifndef OPERATION_BASE_H_
#define OPERATION_BASE_H_

// C/C++ headers
#include <typeindex>
#include <unordered_map>
#include <vector>

// Engine headers
#include "../token.hpp"
#include "types.hpp"

namespace zhetapi {

namespace detail {

// Aliases
using Operation = std::function <Token *(const Targs &)>;
using Signature = std::vector <std::type_index>;
using Overload = std::vector <std::pair <Signature, Operation>>;

// TODO: make const?
extern std::unordered_map <std::string, Operation> universals;
extern std::unordered_map <std::string, Overload> operations;

Token *compute(const std::string&, const Targs &);

std::string gen_signature_str(const Signature &);
std::string overload_catalog(const std::string &);
std::string gen_overload_msg(const Signature &, const std::string &);

Signature gen_signature(const Targs &);
	
// Exceptions
class bad_operation : public std::runtime_error {
public:
	explicit bad_operation(const std::string &str)
		: std::runtime_error(str) {}
};

class bad_overload : public std::runtime_error {
public:
	explicit bad_overload(const std::string &str)
		: std::runtime_error(str) {}
};

class bad_signature : public std::runtime_error {
public:
	explicit bad_signature()
		: std::runtime_error("Nullptr when generating"
			" operation overload.") {}
};

// Macros for operations constructions
#define unary_operation_blank(in, out, blank)				\
	{								\
		{typeid(Operand <in>)},					\
		[](const Targs &args) {					\
			Operand <in> *a = dynamic_cast			\
				<Operand <in> *> (args[0]);		\
									\
			return new Operand <out> (blank);		\
		}							\
	}

#define binary_operation(in1, in2, out, str)				\
	{								\
		{typeid(Operand <in1>), typeid(Operand <in2>)},		\
		[](const Targs &args) {					\
			Operand <in1> *a = dynamic_cast			\
				<Operand <in1> *> (args[0]);		\
			Operand <in2> *b = dynamic_cast			\
				<Operand <in2> *> (args[1]);		\
									\
			return new Operand <out>			\
				(a->get() str b->get());		\
		}							\
	}

#define binary_operation_ftn(in1, in2, out, ftn)			\
	{								\
		{typeid(Operand <in1>), typeid(Operand <in2>)},		\
		[](const Targs &args) {					\
			Operand <in1> *a = dynamic_cast			\
				<Operand <in1> *> (args[0]);		\
			Operand <in2> *b = dynamic_cast			\
				<Operand <in2> *> (args[1]);		\
									\
			return new Operand <out>			\
				(ftn(a->get(), b->get()));		\
		}							\
	}

#define het_binary_operation(in1, in2, out, str)			\
	binary_operation(in1, in2, out, str),				\
	binary_operation(in2, in1, out, str)

#define casted_binary_operation(in1, in2, out, str)			\
	{								\
		{typeid(Operand <in1>), typeid(Operand <in2>)},		\
		[](const Targs &args) {					\
			Operand <in1> *a = dynamic_cast			\
				<Operand <in1> *> (args[0]);		\
			Operand <in2> *b = dynamic_cast			\
				<Operand <in2> *> (args[1]);		\
									\
			return new Operand <out>			\
				(out(a->get()) str out(b->get()));	\
		}							\
	}

#define het_casted_binary_operation(in1, in2, out, str)			\
	casted_binary_operation(in1, in2, out, str),			\
	casted_binary_operation(in2, in1, out, str)

#define casted_binary_operation_set(str)				\
	casted_binary_operation(Z, Z, Z, str),				\
	casted_binary_operation(Q, Q, Q, str),				\
	casted_binary_operation(R, R, R, str),				\
	casted_binary_operation(CmpZ, CmpZ, CmpZ, str),			\
	casted_binary_operation(CmpQ, CmpQ, CmpQ, str),			\
	casted_binary_operation(CmpR, CmpR, CmpR, str),			\
									\
	het_casted_binary_operation(Z, Q, Q, str),			\
	het_casted_binary_operation(Z, R, R, str),			\
	het_casted_binary_operation(Z, CmpZ, CmpZ, str),		\
	het_casted_binary_operation(Z, CmpQ, CmpQ, str),		\
	het_casted_binary_operation(Z, CmpR, CmpR, str),		\
									\
	het_casted_binary_operation(Q, R, R, str),			\
	het_casted_binary_operation(Q, CmpZ, CmpQ, str),		\
	het_casted_binary_operation(Q, CmpQ, CmpQ, str),		\
	het_casted_binary_operation(Q, CmpR, CmpR, str),		\
									\
	het_casted_binary_operation(R, CmpZ, CmpR, str),		\
	het_casted_binary_operation(R, CmpQ, CmpR, str),		\
	het_casted_binary_operation(R, CmpR, CmpR, str),		\
									\
	het_casted_binary_operation(CmpZ, CmpQ, CmpQ, str),		\
	het_casted_binary_operation(CmpZ, CmpR, CmpR, str),		\
									\
	het_casted_binary_operation(CmpQ, CmpR, CmpR, str)		\

#define casted_vector_binary_operation_set(str)				\
	casted_binary_operation(VecZ, VecZ, VecZ, str),			\
	casted_binary_operation(VecQ, VecQ, VecQ, str),			\
	casted_binary_operation(VecR, VecR, VecR, str),			\
	casted_binary_operation(VecCmpZ, VecCmpZ, VecCmpZ, str),	\
	casted_binary_operation(VecCmpQ, VecCmpQ, VecCmpQ, str),	\
	casted_binary_operation(VecCmpR, VecCmpR, VecCmpR, str),	\
									\
	het_casted_binary_operation(VecZ, VecQ, VecQ, str),		\
	het_casted_binary_operation(VecZ, VecR, VecR, str),		\
	het_casted_binary_operation(VecZ, VecCmpZ, VecCmpZ, str),	\
	het_casted_binary_operation(VecZ, VecCmpQ, VecCmpQ, str),	\
	het_casted_binary_operation(VecZ, VecCmpR, VecCmpR, str),	\
									\
	het_casted_binary_operation(VecQ, VecR, VecR, str),		\
	het_casted_binary_operation(VecQ, VecCmpZ, VecCmpQ, str),	\
	het_casted_binary_operation(VecQ, VecCmpQ, VecCmpQ, str),	\
	het_casted_binary_operation(VecQ, VecCmpR, VecCmpR, str),	\
									\
	het_casted_binary_operation(VecR, VecCmpZ, VecCmpR, str),	\
	het_casted_binary_operation(VecR, VecCmpQ, VecCmpR, str),	\
	het_casted_binary_operation(VecR, VecCmpR, VecCmpR, str),	\
									\
	het_casted_binary_operation(VecCmpZ, VecCmpQ, VecCmpQ, str),	\
	het_casted_binary_operation(VecCmpZ, VecCmpR, VecCmpR, str),	\
									\
	het_casted_binary_operation(VecCmpQ, VecCmpR, VecCmpR, str)	\

#define casted_matrix_binary_operation_set(str)				\
	casted_binary_operation(MatZ, MatZ, MatZ, str),			\
	casted_binary_operation(MatQ, MatQ, MatQ, str),			\
	casted_binary_operation(MatR, MatR, MatR, str),			\
	casted_binary_operation(MatCmpZ, MatCmpZ, MatCmpZ, str),	\
	casted_binary_operation(MatCmpQ, MatCmpQ, MatCmpQ, str),	\
	casted_binary_operation(MatCmpR, MatCmpR, MatCmpR, str),	\
									\
	het_casted_binary_operation(MatZ, MatQ, MatQ, str),		\
	het_casted_binary_operation(MatZ, MatR, MatR, str),		\
	het_casted_binary_operation(MatZ, MatCmpZ, MatCmpZ, str),	\
	het_casted_binary_operation(MatZ, MatCmpQ, MatCmpQ, str),	\
	het_casted_binary_operation(MatZ, MatCmpR, MatCmpR, str),	\
									\
	het_casted_binary_operation(MatQ, MatR, MatR, str),		\
	het_casted_binary_operation(MatQ, MatCmpZ, MatCmpQ, str),	\
	het_casted_binary_operation(MatQ, MatCmpQ, MatCmpQ, str),	\
	het_casted_binary_operation(MatQ, MatCmpR, MatCmpR, str),	\
									\
	het_casted_binary_operation(MatR, MatCmpZ, MatCmpR, str),	\
	het_casted_binary_operation(MatR, MatCmpQ, MatCmpR, str),	\
	het_casted_binary_operation(MatR, MatCmpR, MatCmpR, str),	\
									\
	het_casted_binary_operation(MatCmpZ, MatCmpQ, MatCmpQ, str),	\
	het_casted_binary_operation(MatCmpZ, MatCmpR, MatCmpR, str),	\
									\
	het_casted_binary_operation(MatCmpQ, MatCmpR, MatCmpR, str)	\

#define scalar_binary_operation_set(str)				\
	binary_operation(Z, Z, B, str),					\
	binary_operation(Q, Q, B, str),					\
	binary_operation(R, R, B, str),					\
	het_binary_operation(Z, Q, B, str),				\
	het_binary_operation(Z, R, B, str),				\
	het_binary_operation(Q, R, B, str)

}

}

#endif