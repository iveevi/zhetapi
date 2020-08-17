#ifndef BARN_H_
#define BARN_H_

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <tuple>
#include <typeindex>
#include <typeinfo>

#include <vector.h>
#include <matrix.h>
#include <operand.h>
#include <operation.h>
#include <rational.h>
#include <token.h>
#include <vtable.h>
#include <complex.h>

#define AB_OP(str, A, B, O)									\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(operand <A>),					\
					typeid(operand <B>)					\
				}								\
			},									\
												\
			new operation <O> {							\
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

#define ABH_OP(str, A, B, O)									\
	AB_OP(str, A, B, O)									\
	AB_OP(str, B, A, O)

#define AB_OP_FTR(str, A, B, O, ftr)								\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(operand <A>),					\
					typeid(operand <B>)					\
				}								\
			},									\
												\
			new operation <O> {							\
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

#define ABH_OP_FTR(str, A, B, O, ftr)								\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {					\
					typeid(operand <A>),					\
					typeid(operand <B>)					\
				}								\
			},									\
												\
			new operation <O> {							\
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
			new operation <O> {							\
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

#define AB_OP_SET(str)										\
	AB_OP(str, Z, Z, Z);									\
	AB_OP(str, R, R, R);									\
	AB_OP(str, Q, Q, Q);									\
	AB_OP(str, CR, CR, CR);									\
	AB_OP(str, CQ, CQ, CQ);									\
												\
	ABH_OP(str, R, Z, R);									\
	ABH_OP_FTR(str, Z, Q, Q, Q(a->get()) str b->get());					\
	ABH_OP(str, R, Q, R);									\
	ABH_OP(str, R, CR, CR);									\
	ABH_OP_FTR(str, R, CQ, CR, CR(a->get() str R(b->get().real()), R(b->get().imag())));	\
	ABH_OP_FTR(str, Z, CR, CR, CR(a->get() str b->get().real(), b->get().imag()));		\
	ABH_OP_FTR(str, Z, CQ, CQ, CQ(a->get() str b->get().real(), b->get().imag()));		\
	ABH_OP_FTR(str, Q, CR, CR, CR(R(a->get()) str b->get().real(), b->get().imag()));	\
	ABH_OP_FTR(str, Q, CQ, CQ, CQ(a->get() str b->get().real(), b->get().imag()));		\
	ABH_OP_FTR(str, CR, CQ, CR, CR(a->get().real() str (R) b->get().real(),			\
		a->get().imag() str (R) b->get().imag()));

#define AU_OP(str, I, O)									\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {typeid(operand <I>)}		\
			},									\
												\
			new operation <O> {							\
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

#define AU_OP_FTR(str, I, O, ftr)								\
	ops.push_back({										\
			{									\
				std::string(#str),						\
				std::vector <std::type_index> {typeid(operand <I>)}		\
			},									\
												\
			new operation <O> {							\
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
 * @brief Represents the working
 * space of a [zhetapi] function
 * or application; the sets of
 * integer, real, complex, Rational,
 * vector and Matrix fields.
 *
 * @tparam R The type of a scalar
 * real value. Defaults to
 * [long double].
 *
 * @tparam Z The type of a scalar
 * integer value (used in Rationals).
 * Defaults to [long long int].
 */
template <class R = long double, class Z = long long int>
class barn {
public:
	using Q = Rational <Z>;

	using CR = Complex <R>;
	using CQ = Complex <Q>;

	using MR = Matrix <R>;
	using MQ = Matrix <Q>;

	using MCR = Matrix <CR>;
	using MCQ = Matrix <CQ>;

	using VR = Vector <R>;
	using VQ = Vector <Q>;

	using VCR = Vector <CR>;
	using VCQ = Vector <CQ>;

	using opd_z = operand <Z>;
	using opd_r = operand <R>;
	using opd_q = operand <Q>;
	using opd_cr = operand <CR>;
	using opd_cq = operand <CQ>;

	using ID = std::pair <std::string, std::vector <std::type_index>>;
private:
	vtable <Z> z;
	vtable <R> r;
	vtable <Q> q;

	vtable <CR> rc;
	vtable <CQ> qc;

	vtable <MR> rm;
	vtable <MQ> qm;

	// ftable <R, Z> maps;

	std::vector <std::pair <ID, token *>> ops;
public:
	barn();

	void put_z(const Z &);
	void put_r(const R &);
	void put_q(const Q &);
	void put_rc(const CR &);
	void put_qc(const CQ &);
	void put_rm(const MR &);
	void put_qm(const MQ &);

	token *mkop(const std::string &, const std::vector <std::type_index> &);
	token *value(const std::string &, const std::vector <std::type_index> &,
			const std::vector <token *> &) const;

	void print();
};

//////////////////////////////////////////
// Constructors
//////////////////////////////////////////

template <class R, class Z>
barn <R, Z> ::barn() : z(), r(), q(), rc(), qc(), rm(), qm()
{
	//////////////////////////////////////////
	// Real Scalar Arithemtic
	//////////////////////////////////////////
	
	// Addition and Subtraction
	AB_OP_SET(+);
	AB_OP_SET(-);

	// Multiplication
	AB_OP(*, Z, Z, Z);
	AB_OP(*, R, R, R);
	AB_OP(*, Q, Q, Q);
	AB_OP(*, CR, CR, CR);
	AB_OP(*, CQ, CQ, CQ);

	ABH_OP(*, R, Z, R);
	ABH_OP(*, Z, Q, Q);
	ABH_OP_FTR(*, R, Q, R, a->get() * (R) b->get());
	ABH_OP(*, R, CR, CR);
	ABH_OP_FTR(*, R, CQ, CR, CR(a->get() * (R) b->get().real(), a->get() * (R) b->get().imag()));
	ABH_OP_FTR(*, Z, CR, CR, CR(a->get() * (R) b->get().real(), a->get() * (R) b->get().imag()));
	ABH_OP_FTR(*, Z, CQ, CQ, CQ((Q) a->get() * b->get().real(), (Q) a->get() * b->get().imag()));
	ABH_OP_FTR(*, Q, CR, CR, CR((R) a->get() * b->get().real(), (R) a->get() * b->get().imag()));
	ABH_OP(*, Q, CQ, CQ);

	// Division
	AB_OP(/, R, R, R);
	AB_OP(/, Q, Q, Q);
	AB_OP(/, CR, CR, CR);
	AB_OP(/, CQ, CQ, CQ);

	ABH_OP(/, R, Z, R);
	ABH_OP(/, Z, Q, Q);
	ABH_OP_FTR(/, R, Q, R, a->get() * (R) b->get());
	ABH_OP(/, R, CR, CR);
	ABH_OP_FTR(/, R, CQ, CR, CR(a->get() * (R) b->get().real(), a->get() * (R) b->get().imag()));
	ABH_OP_FTR(/, Z, CR, CR, CR(a->get() * (R) b->get().real(), a->get() * (R) b->get().imag()));
	ABH_OP_FTR(/, Z, CQ, CQ, CQ((Q) a->get() * b->get().real(), (Q) a->get() * b->get().imag()));
	ABH_OP_FTR(/, Q, CR, CR, CR((R) a->get() * b->get().real(), (R) a->get() * b->get().imag()));
	ABH_OP(/, Q, CQ, CQ);

	// Exponentiation
	AB_OP_FTR(^, Z, Z, Z, (Z) pow(a->get(), b->get()));
	AB_OP_FTR(^, R, R, R, pow(a->get(), b->get()));
	AB_OP_FTR(^, Q, Q, R, pow((R) a->get(), (R) b->get()));
	AB_OP_FTR(^, CR, CR, CR, 1);
	AB_OP_FTR(^, CQ, CQ, CQ, Q(1));

	// Trigonometry
	AU_OP(sin, Z, R);
	AU_OP_FTR(sin, Q, R, sin(R (in->get())));
	AU_OP(sin, R, R);
	
	AU_OP(cos, Z, R);
	AU_OP_FTR(sin, Q, R, cos(R (in->get())));
	AU_OP(cos, R, R);
	
	AU_OP(tan, Z, R);
	AU_OP_FTR(sin, Q, R, tan(R (in->get())));
	AU_OP(tan, R, R);

	// Vector
	AB_OP(+, VQ, VQ, VQ);
	AB_OP(+, VR, VR, VR);
	
	AB_OP(-, VQ, VQ, VQ);
	AB_OP(-, VR, VR, VR);
	
	AB_OP_FTR(., VQ, VQ, Q, inner(a->get(), b->get()));
	AB_OP_FTR(., VR, VR, R, inner(a->get(), b->get()));
	
	AB_OP_FTR(shur, VQ, VQ, VQ, shur(a->get(), b->get()));
	AB_OP_FTR(shur, VR, VR, VR, shur(a->get(), b->get()));

	AU_OP_FTR(transpose, VQ, MQ, in->get().transpose());
	AU_OP_FTR(transpose, VR, MR, in->get().transpose());

	// Matrix
	AB_OP(+, MQ, MQ, MQ);
	AB_OP(+, MR, MR, MR);
	
	AB_OP(-, MQ, MQ, MQ);
	AB_OP(-, MR, MR, MR);
	
	AB_OP_FTR(shur, MQ, MQ, MQ, shur(a->get(), b->get()));
	AB_OP_FTR(shur, MR, MR, MR, shur(a->get(), b->get()));

	AU_OP_FTR(transpose, MQ, MQ, in->get().transpose());
	AU_OP_FTR(transpose, MR, MR, in->get().transpose());
}

template <class R, class Z>
token *barn <R, Z> ::mkop(const std::string &str,
		const std::vector <std::type_index> &types)
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

	if (it != ops.end())
		return it->second->copy();

	return nullptr;
}

template <class R, class Z>
token *barn <R, Z> ::value(const std::string &str,
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

		// cout << "Token Established: " << tptr->str() << endl;
		
		// Constants
		operation <R> *opn_r = dynamic_cast <operation <R> *> (tptr);
		if (opn_r != nullptr)
			return (*opn_r)(vals);

		operation <Z> *opn_z = dynamic_cast <operation <Z> *> (tptr);
		if (opn_z != nullptr)
			return (*opn_z)(vals);

		operation <Q> *opn_q = dynamic_cast <operation <Q> *> (tptr);
		if (opn_q != nullptr)
			return (*opn_q)(vals);

		operation <CR> *opn_cr = dynamic_cast <operation <CR> *> (tptr);
		if (opn_cr != nullptr)
			return (*opn_cr)(vals);

		operation <CQ> *opn_cq = dynamic_cast <operation <CQ> *> (tptr);
		if (opn_cq != nullptr)
			return (*opn_cq)(vals);

		// Vectors
		operation <VQ> *opn_vq = dynamic_cast <operation <VQ> *> (tptr);
		if (opn_vq != nullptr)
			return (*opn_vq)(vals);

		operation <VR> *opn_vr = dynamic_cast <operation <VR> *> (tptr);
		if (opn_vr != nullptr)
			return (*opn_vr)(vals);

		operation <VCQ> *opn_vcq = dynamic_cast <operation <VCQ> *> (tptr);
		if (opn_vcq != nullptr)
			return (*opn_vcq)(vals);

		operation <VCR> *opn_vcr = dynamic_cast <operation <VCR> *> (tptr);
		if (opn_vcr != nullptr)
			return (*opn_vcr)(vals);

		// Matrices
		operation <MQ> *opn_mq = dynamic_cast <operation <MQ> *> (tptr);
		if (opn_mq != nullptr)
			return (*opn_mq)(vals);

		operation <MR> *opn_mr = dynamic_cast <operation <MR> *> (tptr);
		if (opn_mr != nullptr)
			return (*opn_mr)(vals);

		operation <MCQ> *opn_mcq = dynamic_cast <operation <MCQ> *> (tptr);
		if (opn_mcq != nullptr)
			return (*opn_mcq)(vals);

		operation <MCR> *opn_mcr = dynamic_cast <operation <MCR> *> (tptr);
		if (opn_mcr != nullptr)
			return (*opn_mcr)(vals);
	} else {
		std::cout << "Invalid Token (Not Found)" << std::endl;
	}

	return nullptr;
}

template <class R, class Z>
void barn <R, Z> ::print()
{
	std::cout << std::string(50, '=') << std::endl;
	std::cout << "INTEGERS:" << std::endl;
	std::cout << std::string(50, '=') << std::endl;

	z.print();

	std::cout << std::string(50, '=') << std::endl;
	std::cout << "REALS:" << std::endl;
	std::cout << std::string(50, '=') << std::endl;

	r.print();
	
	std::cout << std::string(50, '=') << std::endl;
	std::cout << "RATIONALS:" << std::endl;
	std::cout << std::string(50, '=') << std::endl;

	q.print();
	
	std::cout << std::string(50, '=') << std::endl;
	std::cout << "REAL COMPLEX:" << std::endl;
	std::cout << std::string(50, '=') << std::endl;

	rc.print();

	std::cout << std::string(50, '=') << std::endl;
	std::cout << "RATIONAL COMPLEX:" << std::endl;
	std::cout << std::string(50, '=') << std::endl;

	qc.print();
	
	std::cout << std::string(50, '=') << std::endl;
	std::cout << "REAL MATRICES:" << std::endl;
	std::cout << std::string(50, '=') << std::endl;

	rm.print();
	
	std::cout << std::string(50, '=') << std::endl;
	std::cout << "RATIONAL MATRICES:" << std::endl;
	std::cout << std::string(50, '=') << std::endl;

	qm.print();
}

#endif
