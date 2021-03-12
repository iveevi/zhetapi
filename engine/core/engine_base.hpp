#ifndef ENGINE_BASE_H_
#define ENGINE_BASE_H_

// C/C++ headers
#include <stdexcept>
#include <unordered_map>
#include <typeinfo>
#include <typeindex>
#include <vector>

// Engine headers
#include <token.hpp>

#include <std/combinatorial.hpp>
#include <core/operation.hpp>
#include <core/types.hpp>

// TODO: Fix these macros to accomodate for __table instead of ops
// Instead of a macro use a private method (clearner code and header)
#define __add_binary_operation(str, A, B, O)							\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(Operand <A>),					\
					typeid(Operand <B>)					\
				}								\
			},									\
												\
			new operation {								\
				::std::string(#str),						\
				"$1 (" + ::std::string(#A) + ") " + ::std::string(#str)		\
					+ " $2 (" + ::std::string(#B) + ")",			\
				2,								\
				[&](const ::std::vector <Token *> &ins) {			\
					Operand <A> *a = dynamic_cast <Operand <A> *> (ins[0]);	\
					Operand <B> *b = dynamic_cast <Operand <B> *> (ins[1]);	\
												\
					O out = a->get() str b->get();				\
												\
					return new Operand <O> (out);				\
				}								\
			}									\
	});

#define __add_heterogenous_binary_operation(str, A, B, O)					\
	__add_binary_operation(str, A, B, O)							\
	__add_binary_operation(str, B, A, O)

#define __add_binary_operation_ftr(str, A, B, O, ftr)						\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(Operand <A>),					\
					typeid(Operand <B>)					\
				}								\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#A) + ") " + std::string(#str)		\
					+ " $2 (" + std::string(#B) + ")",			\
				2,								\
				[&](const ::std::vector <Token *> &ins) {			\
					Operand <A> *a = dynamic_cast <Operand <A> *> (ins[0]);	\
					Operand <B> *b = dynamic_cast <Operand <B> *> (ins[1]);	\
												\
					return new Operand <O> (ftr);				\
				}								\
			}									\
	});

#define __add_heterogenous_binary_operation_ftr(str, A, B, O, ftr)				\
	ops.push_back({										\
			{									\
				::std::string(#str),						\
				::std::vector <::std::type_index> {				\
					typeid(Operand <A>),					\
					typeid(Operand <B>)					\
				}								\
			},									\
												\
			new operation {								\
				::std::string(#str),						\
				"$1 (" + ::std::string(#A) + ") " + ::std::string(#str)		\
					+ " $2 (" + ::std::string(#B) + ")",			\
				2,								\
				[&](const ::std::vector <Token *> &ins) {			\
					Operand <A> *a = dynamic_cast <Operand <A> *> (ins[0]);	\
					Operand <B> *b = dynamic_cast <Operand <B> *> (ins[1]);	\
												\
					return new Operand <O> (ftr);				\
				}								\
			}									\
	});											\
												\
	ops.push_back({										\
			{									\
				::std::string(#str),						\
				::std::vector <::std::type_index> {				\
					typeid(Operand <B>),					\
					typeid(Operand <A>)					\
				}								\
			},									\
												\
			new operation {								\
				::std::string(#str),						\
				"$1 (" + ::std::string(#B) + ") " + ::std::string(#str)		\
					+ " $2 (" + ::std::string(#A) + ")",			\
				2,								\
				[&](const std::vector <Token *> &ins) {			\
					Operand <A> *a = dynamic_cast <Operand <A> *> (ins[1]);	\
					Operand <B> *b = dynamic_cast <Operand <B> *> (ins[0]);	\
												\
					return new Operand <O> (ftr);				\
				}								\
			}									\
	});

#define __add_binary_operation_set(str)								\
	__add_binary_operation(str, Z, Z, Z);							\
	__add_binary_operation(str, R, R, R);							\
	__add_binary_operation(str, Q, Q, Q);							\
	__add_binary_operation(str, CR, CR, CR);						\
	__add_binary_operation(str, CQ, CQ, CQ);						\
												\
	__add_heterogenous_binary_operation(str, R, Z, R);					\
	__add_heterogenous_binary_operation_ftr(str, Z, Q, Q, Q(a->get()) str b->get());	\
	__add_heterogenous_binary_operation(str, R, Q, R);					\
	__add_heterogenous_binary_operation(str, R, CR, CR);					\
	__add_heterogenous_binary_operation_ftr(str, R, CQ, CR, CR(a->get() str			\
				R(b->get().real()), R(b->get().imag())));			\
	__add_heterogenous_binary_operation_ftr(str, Z, CR, CR, CR(a->get() str			\
				b->get().real(), b->get().imag()));				\
	__add_heterogenous_binary_operation_ftr(str, Z, CQ, CQ, CQ(a->get() str			\
				b->get().real(), b->get().imag()));				\
	__add_heterogenous_binary_operation_ftr(str, Q, CR, CR, CR(R(a->get())			\
				str b->get().real(), b->get().imag()));				\
	__add_heterogenous_binary_operation_ftr(str, Q, CQ, CQ, CQ(a->get() str			\
				b->get().real(), b->get().imag()));				\
	__add_heterogenous_binary_operation_ftr(str, CR, CQ, CR,				\
			CR(a->get().real() str (R) b->get().real(),				\
				a->get().imag() str (R) b->get().imag()));

#define __add_unary_operation(str, I, O)							\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {typeid(Operand <I>)}		\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#I) + ")",					\
				1, 								\
				[&](const std::vector <Token *> &ins) {				\
					Operand <I> *in = dynamic_cast				\
						<Operand <I> *> (ins[0]);			\
												\
					return new Operand <O> (str(in->get()));		\
				}								\
			}									\
	});

#define __add_unary_operation_ftr(str, I, O, ftr)						\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {typeid(Operand <I>)}		\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#I) + ")",					\
				1, 								\
				[&](const std::vector <Token *> &ins) {				\
					Operand <I> *in = dynamic_cast				\
						<Operand <I> *> (ins[0]);			\
												\
					return new Operand <O> (ftr);				\
				}								\
			}									\
	});

namespace zhetapi {

// Symbol table alias
template <class T>
using symtab = std::unordered_map <std::string, T>;

// Other aliases
using signature = std::vector <std::type_index>;
using overloads = std::vector <std::pair <signature, Token *>>;

// Class for storing overloads
class engine_base {
	symtab <overloads>	__overloads;

	// Private methods
	std::string gen_overload_msg(const signature &, const std::string &);
	signature gen_signature(const std::vector <Token *> &);
public:
	engine_base();
	engine_base(const engine_base &);

	engine_base &operator=(const engine_base &);

	~engine_base();

	Token *compute(const std::string &, const std::vector <Token *> &);

	std::string overload_catalog(const std::string &);
	
	// Exceptions
	class unknown_op_overload : public std::runtime_error {
	public:
		explicit unknown_op_overload(const std::string &str)
				: std::runtime_error(str) {}
	};
};

}

#endif
