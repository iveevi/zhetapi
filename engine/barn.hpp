#ifndef BARN_H_
#define BARN_H_

// C/C++ headers
#include <ctime>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

// Engine headers
#include <core/algorithm.hpp>
#include <function.hpp>

#include <complex.hpp>
#include <matrix.hpp>
#include <operand.hpp>
#include <rational.hpp>
#include <registration.hpp>
#include <token.hpp>
#include <vector.hpp>
#include <variable.hpp>

#include <core/types.hpp>
#include <core/operation.hpp>

#include <std/combinatorial.hpp>

namespace zhetapi {

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

class Function;
class algorithm;

/**
 * @brief Represents the working space of a [zhetapi] function or
 * application; the sets of integer, real, complex, rational, vector and
 * matrix fields.
 *
 * @tparam R The type of a scalar real value. Defaults to [long double].
 *
 * @tparam Z The type of a scalar integer value (used in Rationals).
 * Defaults to [long long int].
 */
class Barn {
public:
	__TYPEDEFS__
	
	using ID = std::pair <std::string, std::vector <std::type_index>>;

	using signature = std::vector <std::type_index>;

	// using __loc_table = ::std::unordered_map <::std::string, ::std::string>;
private:
	//ftable <T, U> fstack;

	std::vector <std::pair <ID, Token *>> ops;

	std::unordered_map <std::string, Variable>		__var_table;
	std::unordered_map <std::string, Function>		__ftr_table;

	std::unordered_map <std::string, Registrable>		__reg_table;
	std::unordered_map <std::string, algorithm>		__alg_table;

	mutable std::unordered_map <std::string, std::vector <std::pair <signature, Token *>>> table;
	
	// __loc_table lt;
public:
	Barn();
	Barn(const Barn &);

	~Barn();

	Barn &operator=(const Barn &);

	bool present(const std::string &) const;

	void put(Variable);
	void put(Function);
	void put(Registrable);
	void put(algorithm);

	/* template <class A>
	void put(const ::std::string &, A); */

	void put(Token *, const std::string &);

	Variable &retrieve_variable(const std::string &);
	Function &retrieve_function(const std::string &);

	Token *get(const std::string &);

	Token *compute(const std::string &, const std::vector <Token *> &) const;
	
	std::string overloads(const std::string &) const;

	void list() const;
	void list_registered(std::string) const;

	void print(bool = false) const;

	// Exceptions
	class unknown_operation_overload {
		std::string	__str;
	public:
		explicit unknown_operation_overload(const std::string &str)
				: __str(str) {}

		const std::string &what() const {
			return __str;
		}
	};
};

}

#endif
