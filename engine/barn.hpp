#ifndef BARN_H_
#define BARN_H_

// C/C++ headers
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>

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
	private:
		vtable <R> v_stack_r;
		vtable <Q> v_stack_q;
		vtable <Z> v_stack_z;

		vtable <CQ> v_stack_cq;
		vtable <CR> v_stack_cr;

		vtable <MQ> v_stack_mq;
		vtable <MR> v_stack_mr;

		std::vector <std::pair <ID, token *>> ops;
	public:
		Barn();
		Barn(const Barn &);

		~Barn();

		/*
		 * Place a variable of type A into its appropriate stack. Made
		 * for the user.
		 */
		template <class A>
		void put(Variable <A>);

		/*
		 * Retrieve a variable of type A from the appropriate stack. Has
		 * a chance of throwing an exception. Made for the user.
		 */
		template <class A>
		Variable <A> retrieve(const std::string &);

		token *get(const std::string &);

		token *value(const std::string &, const std::vector <std::type_index> &,
				const std::vector <token *> &) const;

		void print();
	};

	//////////////////////////////////////////
	// Constructors
	//////////////////////////////////////////

	template <class T, class U>
	Barn <T, U> ::Barn() : v_stack_z(), v_stack_r(), v_stack_q(), v_stack_cq(),
		v_stack_cr(), v_stack_mq(), v_stack_mr()
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
	}

	template <class T, class U>
	Barn <T, U> ::Barn(const Barn <T, U> &other) :
		v_stack_r(other.v_stack_r), v_stack_q(other.v_stack_q),
		v_stack_z(other.v_stack_z), v_stack_cq(other.v_stack_cq),
		v_stack_cr(other.v_stack_cr), v_stack_mr(other.v_stack_mr),
		v_stack_mq(other.v_stack_mq)
	{
		for (auto pr : other.ops)
			ops.push_back({pr.first, pr.second->copy()});
	}

	template <class T, class U>
	Barn <T, U> ::~Barn()
	{
		for (auto pr : ops)
			delete pr.second;
	}

	template <class T, class U>
	template <class A>
	void Barn <T, U> ::put(Variable <A> var)
	{
		if (typeid(A) == typeid(Z))
			v_stack_z.insert(var);
		if (typeid(A) == typeid(R))
			v_stack_r.insert(var);
		if (typeid(A) == typeid(Q))
			v_stack_q.insert(var);
		
		if (typeid(A) == typeid(CR))
			v_stack_cr.insert(var);
		if (typeid(A) == typeid(CQ))
			v_stack_cq.insert(var);
		
		if (typeid(A) == typeid(VR))
			v_stack_mr.insert(var);
		if (typeid(A) == typeid(VQ))
			v_stack_mq.insert(var);
	}

	template <class T, class U>
	template <class A>
	Variable <A> Barn <T, U> ::retrieve(const std::string &str)
	{
		if (typeid(A) == typeid(Z))
			v_stack_z.get(str);
		if (typeid(A) == typeid(R))
			v_stack_r.get(str);
		if (typeid(A) == typeid(Q))
			v_stack_q.get(str);
		
		if (typeid(A) == typeid(CR))
			v_stack_cr.get(str);
		if (typeid(A) == typeid(CQ))
			v_stack_cq.get(str);
		
		if (typeid(A) == typeid(VR))
			v_stack_mr.get(str);
		if (typeid(A) == typeid(VQ))
			v_stack_mq.get(str);
	}

	template <class T, class U>
	token *Barn <T, U> ::get(const std::string &str)
	{
		if (v_stack_z.contains(str))
			return new operand <Z> (v_stack_z.get(str).get());
		if (v_stack_r.contains(str))
			return new operand <R> (v_stack_r.get(str).get());
		if (v_stack_q.contains(str))
			return new operand <Q> (v_stack_q.get(str).get());
		
		if (v_stack_cr.contains(str))
			return new operand <CR> (v_stack_cr.get(str).get());
		if (v_stack_cq.contains(str))
			return new operand <CQ> (v_stack_cq.get(str).get());
		
		if (v_stack_mr.contains(str))
			return new operand <VR> (v_stack_mr.get(str).get());
		if (v_stack_mq.contains(str))
			return new operand <VQ> (v_stack_mq.get(str).get());

		return nullptr;
	}

	template <class T, class U>
	token *Barn <T, U> ::value(const std::string &str,
			const std::vector <std::type_index> &types,
			const std::vector <token *> &vals) const
	{
		auto it = ops.end();

		for (auto itr = ops.begin(); itr != ops.end(); itr++) {
			if (itr->first.first == str &&
				itr->first.second.size() == types.size()) {
				bool ps = true;
				
				for (size_t i = 0; i < types.size(); i++) {
					if (types[i] != itr->first.second[i]) {
						ps = false;
						break;
					}
				}

				if (ps) {
					it = itr;
					break;
				}
			}
		}

		if (it != ops.end()) {
			token *tptr = it->second;
			operation *optr = dynamic_cast <operation *> (tptr);
			return (*optr)(vals);
		} else {
			std::cout << "Invalid Token (Not Found)" << std::endl;
		}

		return nullptr;
	}

	template <class T, class U>
	void Barn <T, U> ::print()
	{
		std::cout << std::string(50, '=') << std::endl;
		std::cout << "INTEGERS:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		v_stack_z.print();

		std::cout << std::string(50, '=') << std::endl;
		std::cout << "REALS:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		v_stack_r.print();
		
		std::cout << std::string(50, '=') << std::endl;
		std::cout << "RATIONALS:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		v_stack_q.print();

		std::cout << std::string(50, '=') << std::endl;
		std::cout << "RATIONAL COMPLEX:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		v_stack_cq.print();
		
		std::cout << std::string(50, '=') << std::endl;
		std::cout << "REAL COMPLEX:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		v_stack_cr.print();
		
		std::cout << std::string(50, '=') << std::endl;
		std::cout << "RATIONAL MATRICES:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		v_stack_mq.print();
		
		std::cout << std::string(50, '=') << std::endl;
		std::cout << "REAL MATRICES:" << std::endl;
		std::cout << std::string(50, '=') << std::endl;

		v_stack_mr.print();
	}

}

#endif
