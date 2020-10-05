#ifndef BARN_H_
#define BARN_H_

// C/C++ headers
#include <cmath>
#include <ctime>
#include <functional>
#include <iostream>
#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

// Engine headers
#include <std_combinatorial.hpp>
#include <complex.hpp>
#include <matrix.hpp>
#include <operand.hpp>
#include <operation.hpp>
#include <rational.hpp>
#include <token.hpp>
#include <types.hpp>
#include <vector.hpp>
#include <vtable.hpp>
#include <ftable.hpp>
#include <variable.hpp>

namespace zhetapi {

#define __add_binary_operation(str, A, B, O)							\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(operand <A>),					\
					typeid(operand <B>)					\
				}								\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#A) + ") " + std::string(#str)		\
					+ " $2 (" + std::string(#B) + ")",			\
				2,								\
				[&](const std::vector <token *> &ins) {				\
					operand <A> *a = dynamic_cast <operand <A> *> (ins[0]);	\
					operand <B> *b = dynamic_cast <operand <B> *> (ins[1]);	\
												\
					return new operand <O> (a->get() str b->get());		\
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
					typeid(operand <A>),					\
					typeid(operand <B>)					\
				}								\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#A) + ") " + std::string(#str)		\
					+ " $2 (" + std::string(#B) + ")",			\
				2,								\
				[&](const std::vector <token *> &ins) {				\
					operand <A> *a = dynamic_cast <operand <A> *> (ins[0]);	\
					operand <B> *b = dynamic_cast <operand <B> *> (ins[1]);	\
												\
					return new operand <O> (ftr);				\
				}								\
			}									\
	});

#define __add_heterogenous_binary_operation_ftr(str, A, B, O, ftr)								\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(operand <A>),					\
					typeid(operand <B>)					\
				}								\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#A) + ") " + std::string(#str)		\
					+ " $2 (" + std::string(#B) + ")",			\
				2,								\
				[&](const std::vector <token *> &ins) {				\
					operand <A> *a = dynamic_cast <operand <A> *> (ins[0]);	\
					operand <B> *b = dynamic_cast <operand <B> *> (ins[1]);	\
												\
					return new operand <O> (ftr);				\
				}								\
			}									\
	});											\
												\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(operand <B>),					\
					typeid(operand <A>)					\
				}								\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#B) + ") " + std::string(#str)		\
					+ " $2 (" + std::string(#A) + ")",			\
				2,								\
				[&](const std::vector <token *> &ins) {				\
					operand <A> *a = dynamic_cast <operand <A> *> (ins[1]);	\
					operand <B> *b = dynamic_cast <operand <B> *> (ins[0]);	\
												\
					return new operand <O> (ftr);				\
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
				std::vector <std::type_index> {typeid(operand <I>)}		\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#I) + ")",					\
				1, 								\
				[&](const std::vector <token *> &ins) {				\
					operand <I> *in = dynamic_cast				\
						<operand <I> *> (ins[0]);			\
												\
					return new operand <O> (str(in->get()));		\
				}								\
			}									\
	});

#define __add_unary_operation_ftr(str, I, O, ftr)						\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {typeid(operand <I>)}		\
			},									\
												\
			new operation {								\
				std::string(#str),						\
				"$1 (" + std::string(#I) + ")",					\
				1, 								\
				[&](const std::vector <token *> &ins) {				\
					operand <I> *in = dynamic_cast				\
						<operand <I> *> (ins[0]);			\
												\
					return new operand <O> (ftr);				\
				}								\
			}									\
	});

	template <class T, class U>
	class Function;

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
	template <class T = long double, class U = long long int>
	class Barn {
	public:
		__TYPEDEFS__
		
		using ID = std::pair <std::string, std::vector <std::type_index>>;

		using signature = std::vector <std::type_index>;
	private:
		vtable <T, U> vstack;
		ftable <T, U> fstack;

		std::vector <std::pair <ID, token *>> ops;

		mutable std::unordered_map <std::string, std::vector <std::pair <signature, token *>>> table;
	public:
		Barn();
		Barn(const Barn &);

		~Barn();

		Barn &operator=(const Barn &);

		bool present(const std::string &) const;

		void put(Variable <T, U>);
		void put(Function <T, U>);

		template <class A>
		void put(const std::string &, A);

		Variable <T, U> retrieve_variable(const std::string &);
		Function <T, U> retrieve_function(const std::string &);

		token *get(const std::string &);

		token *compute(const std::string &, const std::vector <token *> &) const;
		
		std::string overloads(const std::string &) const;

		void list() const;
		void print(bool = false) const;

		// Exceptions
		class unknown_operation_overload_exception {
			std::string __str;
		public:
			unknown_operation_overload_exception(const std::string &str) : __str(str) {}

			const std::string &what() const {
				return __str;
			}
		};
	};

	//////////////////////////////////////////
	// Constructors
	//////////////////////////////////////////

	template <class T, class U>
	Barn <T, U> ::Barn() : vstack(), fstack()
	{
		//////////////////////////////////////////
		// Real Scalar Arithemtic
		//////////////////////////////////////////
		
		// Addition and Subtraction
		__add_binary_operation_set(+);
		__add_binary_operation_set(-);

		// Multiplication
		__add_binary_operation(*, Z, Z, Z);
		__add_binary_operation(*, R, R, R);
		__add_binary_operation(*, Q, Q, Q);
		__add_binary_operation(*, CR, CR, CR);
		__add_binary_operation(*, CQ, CQ, CQ);

		__add_heterogenous_binary_operation(*, R, Z, R);
		__add_heterogenous_binary_operation(*, Z, Q, Q);
		__add_heterogenous_binary_operation_ftr(*, R, Q, R, a->get() * (R)
				b->get());
		__add_heterogenous_binary_operation(*, R, CR, CR);
		__add_heterogenous_binary_operation_ftr(*, R, CQ, CR, CR(a->get() * (R)
					b->get().real(), a->get() * (R)
					b->get().imag()));
		__add_heterogenous_binary_operation_ftr(*, Z, CR, CR, CR(a->get() * (R)
					b->get().real(), a->get() * (R)
					b->get().imag()));
		__add_heterogenous_binary_operation_ftr(*, Z, CQ, CQ, CQ((Q) a->get() *
					b->get().real(), (Q) a->get() *
					b->get().imag()));
		__add_heterogenous_binary_operation_ftr(*, Q, CR, CR, CR((R) a->get() *
					b->get().real(), (R) a->get() *
					b->get().imag()));
		__add_heterogenous_binary_operation(*, Q, CQ, CQ);

		// Division
		__add_binary_operation(/, R, R, R);
		__add_binary_operation(/, Q, Q, Q);
		__add_binary_operation(/, CR, CR, CR);
		__add_binary_operation(/, CQ, CQ, CQ);
		
		__add_binary_operation_ftr(/, Z, Z, Q, ((Q) a->get()) / ((Q) b->get()));

		__add_heterogenous_binary_operation(/, R, Z, R);
		__add_heterogenous_binary_operation_ftr(/, Z, Q, Q, ((Q)
					a->get()) / ((Q) b->get()));
		__add_heterogenous_binary_operation_ftr(/, R, Q, R, a->get() * (R)
				b->get());
		__add_heterogenous_binary_operation(/, R, CR, CR);
		__add_heterogenous_binary_operation_ftr(/, R, CQ, CR, CR(a->get() * (R)
					b->get().real(), a->get() * (R)
					b->get().imag()));
		__add_heterogenous_binary_operation_ftr(/, Z, CR, CR, CR(a->get() * (R)
					b->get().real(), a->get() * (R)
					b->get().imag()));
		__add_heterogenous_binary_operation_ftr(/, Z, CQ, CQ, CQ((Q) a->get() *
					b->get().real(), (Q) a->get() *
					b->get().imag()));
		__add_heterogenous_binary_operation_ftr(/, Q, CR, CR, CR((R) a->get() *
					b->get().real(), (R) a->get() *
					b->get().imag()));
		__add_heterogenous_binary_operation(/, Q, CQ, CQ);

		// Exponentiation
		__add_binary_operation_ftr(^, Z, Z, Z, (Z) pow(a->get(), b->get()));
		__add_binary_operation_ftr(^, R, R, R, pow(a->get(), b->get()));
		__add_binary_operation_ftr(^, R, Z, R, pow(a->get(), b->get()));
		__add_binary_operation_ftr(^, Q, Q, R, pow((R) a->get(), (R) b->get()));
		__add_binary_operation_ftr(^, CR, CR, CR, 1);
		__add_binary_operation_ftr(^, CQ, CQ, CQ, Q(1));

		// Trigonometry
		__add_unary_operation(sin, Z, R);
		__add_unary_operation_ftr(sin, Q, R, sin(R (in->get())));
		__add_unary_operation(sin, R, R);
		
		__add_unary_operation(cos, Z, R);
		__add_unary_operation_ftr(cos, Q, R, cos(R (in->get())));
		__add_unary_operation(cos, R, R);
		
		__add_unary_operation(tan, Z, R);
		__add_unary_operation_ftr(tan, Q, R, tan(R (in->get())));
		__add_unary_operation(tan, R, R);
		
		__add_unary_operation_ftr(csc, Z, R, 1/sin(in->get()));
		__add_unary_operation_ftr(csc, Q, R, 1/sin(R (in->get())));
		__add_unary_operation_ftr(csc, R, R, 1/sin(in->get()));
		
		__add_unary_operation_ftr(sec, Z, R, 1/cos(in->get()));
		__add_unary_operation_ftr(sec, Q, R, 1/cos(R (in->get())));
		__add_unary_operation_ftr(sec, R, R, 1/cos(in->get()));
		
		__add_unary_operation_ftr(cot, Z, R, 1/tan(in->get()));
		__add_unary_operation_ftr(cot, Q, R, 1/tan(R (in->get())));
		__add_unary_operation_ftr(cot, R, R, 1/tan(in->get()));

		// Hyperbolic Functions
		__add_unary_operation(sinh, Z, R);
		__add_unary_operation_ftr(sinh, Q, R, sinh(R (in->get())));
		__add_unary_operation(sinh, R, R);
		
		__add_unary_operation(cosh, Z, R);
		__add_unary_operation_ftr(cosh, Q, R, cosh(R (in->get())));
		__add_unary_operation(cosh, R, R);
		
		__add_unary_operation(tanh, Z, R);
		__add_unary_operation_ftr(tanh, Q, R, tanh(R (in->get())));
		__add_unary_operation(tanh, R, R);
		
		__add_unary_operation_ftr(csch, Z, R, 1/sinh(in->get()));
		__add_unary_operation_ftr(csch, Q, R, 1/sinh(R (in->get())));
		__add_unary_operation_ftr(csch, R, R, 1/sinh(in->get()));
		
		__add_unary_operation_ftr(sech, Z, R, 1/cosh(in->get()));
		__add_unary_operation_ftr(sech, Q, R, 1/cosh(R (in->get())));
		__add_unary_operation_ftr(sech, R, R, 1/cosh(in->get()));
		
		__add_unary_operation_ftr(coth, Z, R, 1/tanh(in->get()));
		__add_unary_operation_ftr(coth, Q, R, 1/tanh(R (in->get())));
		__add_unary_operation_ftr(coth, R, R, 1/tanh(in->get()));

		// Logarithms
		__add_unary_operation_ftr(ln, Z, R, log(R (in->get())));
		__add_unary_operation_ftr(ln, Q, R, log(R (in->get())));
		__add_unary_operation_ftr(ln, R, R, log(R (in->get())));
		
		__add_unary_operation_ftr(log, Z, R, log10(R (in->get())));
		__add_unary_operation_ftr(log, Q, R, log10(R (in->get())));
		__add_unary_operation_ftr(log, R, R, log10(R (in->get())));
		
		__add_binary_operation_ftr(log, Z, Z, R, log(R (b->get()))/log(R (a->get())));
		__add_binary_operation_ftr(log, Q, Q, R, log(R (b->get()))/log(R (a->get())));
		__add_binary_operation_ftr(log, R, R, R, log(R (b->get()))/log(R (a->get())));

		__add_heterogenous_binary_operation_ftr(log, Z, R, R, log(R (b->get()))/log(R (a->get())));
		__add_heterogenous_binary_operation_ftr(log, Z, Q, R, log(R (b->get()))/log(R (a->get())));
		__add_heterogenous_binary_operation_ftr(log, Q, R, R, log(R (b->get()))/log(R (a->get())));
		
		__add_unary_operation_ftr(lg, Z, R, log(R (in->get()))/log(2));
		__add_unary_operation_ftr(lg, Q, R, log(R (in->get()))/log(2));
		__add_unary_operation_ftr(lg, R, R, log(R (in->get()))/log(2));
		
		//////////////////////////////////////////
		// Vector operations
		//////////////////////////////////////////

		// Addition
		__add_binary_operation(+, VZ, VZ, VZ);
		__add_binary_operation(+, VQ, VQ, VQ);
		__add_binary_operation(+, VR, VR, VR);
		
		// Subtraction
		__add_binary_operation(-, VZ, VZ, VZ);
		__add_binary_operation(-, VQ, VQ, VQ);
		__add_binary_operation(-, VR, VR, VR);

		// Scalar multiplication
		__add_binary_operation_ftr(*, Z, VZ, VZ, (a->get()) * b->get());
		__add_binary_operation_ftr(*, Z, VQ, VQ, ((Q) a->get()) * b->get());
		
		// Dot product
		__add_binary_operation_ftr(., VZ, VZ, Z, inner(a->get(), b->get()));
		__add_binary_operation_ftr(., VQ, VQ, Q, inner(a->get(), b->get()));
		__add_binary_operation_ftr(., VR, VR, R, inner(a->get(), b->get()));
		
		// Shur product
		__add_binary_operation_ftr(shur, VQ, VQ, VQ, shur(a->get(), b->get()));
		__add_binary_operation_ftr(shur, VR, VR, VR, shur(a->get(), b->get()));

		// Transposition
		__add_unary_operation_ftr(transpose, VQ, MQ, in->get().transpose());
		__add_unary_operation_ftr(transpose, VR, MR, in->get().transpose());
		
		//////////////////////////////////////////
		// Matrix operations
		//////////////////////////////////////////

		__add_binary_operation(+, MQ, MQ, MQ);
		__add_binary_operation(+, MR, MR, MR);
		
		__add_binary_operation(-, MQ, MQ, MQ);
		__add_binary_operation(-, MR, MR, MR);
		
		__add_binary_operation_ftr(shur, MQ, MQ, MQ, shur(a->get(), b->get()));
		__add_binary_operation_ftr(shur, MR, MR, MR, shur(a->get(), b->get()));

		__add_unary_operation_ftr(transpose, MQ, MQ, in->get().transpose());
		__add_unary_operation_ftr(transpose, MR, MR, in->get().transpose());

		// Other Linear Algebra
		__add_binary_operation(*, MQ, VQ, MQ);
		__add_binary_operation(*, VQ, MQ, MQ);

		//////////////////////////////////////////
		// Combinatorial Operations
		//////////////////////////////////////////
		__add_unary_operation_ftr(!, Z, Z,
				utility::integral_factorial(in->get())); 
		
		__add_binary_operation_ftr(binom, Z, Z, Z,
				utility::integral_binom(a->get(), b->get()));

		//////////////////////////////////////////
		// API functions
		//////////////////////////////////////////

		//////////////////////////////////////////
		// Transfer into table
		//////////////////////////////////////////

		for (auto pr : ops) {
			operation *opn = dynamic_cast <operation *> (pr.second);

			if (table.count(pr.first.first))
				table[pr.first.first].push_back({pr.first.second, pr.second});
			else
				table.insert(std::pair <std::string, std::vector <std::pair <signature, token *>>> {pr.first.first, std::vector <std::pair <signature, token *>> {{pr.first.second, pr.second}}});
		}
	}

	template <class T, class U>
	Barn <T, U> ::Barn(const Barn <T, U> &other) : vstack(other.vstack), fstack(other.fstack)
	{
		for (auto pr : other.ops)
			ops.push_back({pr.first, pr.second->copy()});
		
		for (auto pr : ops) {
			operation *opn = dynamic_cast <operation *> (pr.second);

			if (table.count(pr.first.first))
				table[pr.first.first].push_back({pr.first.second, pr.second});
			else
				table.insert(std::pair <std::string, std::vector <std::pair <signature, token *>>> {pr.first.first, std::vector <std::pair <signature, token *>> {{pr.first.second, pr.second}}});
		}
	}
	
	template <class T, class U>
	Barn <T, U> &Barn <T, U> ::operator=(const Barn <T, U> &other)
	{
		if (this != &other) {
			vstack = other.vstack;
			fstack = other.fstack;

			for (auto pr : other.ops)
				ops.push_back({pr.first, pr.second->copy()});
			
			for (auto pr : ops) {
				operation *opn = dynamic_cast <operation *> (pr.second);

				if (table.count(pr.first.first))
					table[pr.first.first].push_back({pr.first.second, pr.second});
				else
					table.insert(std::pair <std::string, std::vector <std::pair <signature, token *>>> {pr.first.first, std::vector <std::pair <signature, token *>> {{pr.first.second, pr.second}}});
			}
		}

		return *this;
	}

	template <class T, class U>
	Barn <T, U> ::~Barn()
	{
		for (auto pr : ops)
			delete pr.second;
	}

	template <class T, class U>
	bool Barn <T, U> ::present(const std::string &str) const
	{
		auto itr = std::find_if(ops.begin(), ops.end(), [&](const std::pair <ID, token *> &pr) {
			return pr.first.first == str;
		});

		return itr != ops.end();
	}

	template <class T, class U>
	void Barn <T, U> ::put(Variable <T, U> var)
	{
		if (vstack.contains(var.symbol()))
			vstack.get(var.symbol()) = var;
		else
			vstack.insert(var);
	}

	template <class T, class U>
	template <class A>
	void Barn <T, U> ::put(const std::string &str, A x)
	{
		put(Variable <T, U> {str, x});
	}

	template <class T, class U>
	Variable <T, U> Barn <T, U> ::retrieve_variable(const std::string &str)
	{
		return vstack.get(str);
	}

	template <class T, class U>
	token *Barn <T, U> ::get(const std::string &str)
	{
		if (vstack.contains(str))
			return (vstack.get(str).get())->copy();
		
		if (fstack.contains(str))
			return (fstack.get(str)).copy();

		return nullptr;
	}
	
	template <class T, class U>
	token *Barn <T, U> ::compute(const std::string &str, const std::vector <token *> &vals) const
	{
		std::vector <std::type_index> sig;

		for (token *tptr : vals)
			sig.push_back(typeid(*tptr));

		token *tptr = nullptr;

		std::vector <std::pair <signature, token *>> *siglist = &table[str];

		size_t sz = sig.size();
		for (auto itr = siglist->begin(); itr != siglist->end(); itr++) {
			if (itr->first.size() == sz) {
				bool ps = true;
				
				for (size_t i = 0; i < sz; i++) {					
					if (sig[i] != itr->first[i]) {
						ps = false;

						break;
					}
				}

				if (ps) {
					tptr = itr->second;
					
					break;
				}
			}
		}
		
		if (tptr) {
			operation *optr = dynamic_cast <operation *> (tptr);

			return (*optr)(vals);
		} else {
			std::ostringstream oss;

			oss << "Unknown overload (";

			for (size_t i = 0; i < sig.size(); i++) {
				oss << types <T, U> ::symbol(sig[i]);
				
				if (i < sig.size() - 1)
					oss << ", ";
			}

			oss << ") for operation \"" << str << "\". " <<
				overloads(str);

			throw unknown_operation_overload_exception(oss.str());
		}

		return nullptr;
	}

	template <class T, class U>
	std::string Barn <T, U> ::overloads(const std::string &str) const
	{
		auto it = ops.end();

		std::vector <std::vector <std::type_index>> loads;
		for (auto itr = ops.begin(); itr != ops.end(); itr++) {
			if (itr->first.first == str)
				loads.push_back(itr->first.second);
		}

		std::ostringstream oss;

		oss << "Available overloads for \"" << str << "\": {";

		for (size_t k = 0; k < loads.size(); k++) {
			oss << "(";

			for (size_t i = 0; i < loads[k].size(); i++) {
				oss << types <T, U> ::symbol(loads[k][i]);
				
				if (i < loads[k].size() - 1)
					oss << ", ";
			}

			oss << ")";

			if (k < loads.size() - 1)
				oss << ", ";
		}

		oss << "}";

		return oss.str();
	}

	template <class T, class U>
	void Barn <T, U> ::list() const
	{
		std::vector <Variable <T, U>> v = vstack.list();

		if (v.size()) {
			std::cout << "\tVariables:" << std::endl;
			for (auto var : v)
				std::cout << "\t\t" << var << std::endl;
		}

		std::vector <Function <T, U>> f = fstack.list();
		
		if (f.size()) {
			std::cout << "\tFunctions:" << std::endl;
			for (auto ftn : f)
				std::cout << "\t\t" << ftn << std::endl;
		}
	}

	template <class T, class U>
	void Barn <T, U> ::print(bool show_ops) const
	{
		std::cout << std::string(50, '=') << std::endl;
		std::cout << "VARIABLES" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		vstack.print();

		std::cout << std::string(50, '=') << std::endl;
		std::cout << "FUNCTIONS:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		fstack.print();

		if (show_ops) {
			for (auto pr : ops) {
				std::cout << "op: " << pr.second->str() << " @ " <<
					pr.second << std::endl;
			}

			std::cout << "######################################################" << std::endl;

			for (auto itr : table) {
				std::cout << "Str: " << itr.first << " @ Size: " << itr.second.size() << " @ 2nd Size: " << table[itr.first].size() << std::endl;

				for (auto pr : itr.second)
					std::cout << "\t" << pr.second->str() << std::endl;
			}
		}
	}

}

#endif
