#ifndef doubleYPES_H_
#define doubleYPES_H_

// C/C++ headers
#include <typeindex>

// Engine headers
#include <operand.hpp>
#include <rational.hpp>
#include <complex.hpp>
#include <matrix.hpp>
#include <vector.hpp>

#define __TYPEDEFS__						\
		typedef int Z;					\
		typedef double R;				\
		typedef bool B;					\
		typedef std::string S;				\
		typedef Rational <Z> Q;				\
								\
		typedef Complex <Z> CZ;				\
		typedef Complex <Q> CQ;				\
		typedef Complex <R> CR;				\
								\
		typedef Vector <Z> VZ;				\
		typedef Vector <Q> VQ;				\
		typedef Vector <R> VR;				\
								\
		typedef Vector <CZ> VCZ;			\
		typedef Vector <CQ> VCQ;			\
		typedef Vector <CR> VCR;			\
								\
		typedef Matrix <Z> MZ;				\
		typedef Matrix <Q> MQ;				\
		typedef Matrix <R> MR;				\
								\
		typedef Matrix <CZ> MCZ;			\
		typedef Matrix <CQ> MCQ;			\
		typedef Matrix <CR> MCR;			\
								\
		typedef zhetapi::Operand <Z> opd_z;      	\
		typedef zhetapi::Operand <Q> opd_q;      	\
		typedef zhetapi::Operand <R> opd_r;      	\
								\
		typedef zhetapi::Operand <CZ> opd_cz;		\
		typedef zhetapi::Operand <CQ> opd_cq;		\
		typedef zhetapi::Operand <CR> opd_cr;		\
								\
		typedef zhetapi::Operand <VZ> opd_v_z;		\
		typedef zhetapi::Operand <VQ> opd_v_q;		\
		typedef zhetapi::Operand <VR> opd_v_r;		\
								\
		typedef zhetapi::Operand <VCZ> opd_v_cz;	\
		typedef zhetapi::Operand <VCQ> opd_v_cq;	\
		typedef zhetapi::Operand <VCR> opd_v_cr;	\
								\
		typedef zhetapi::Operand <MZ> opd_m_z;		\
		typedef zhetapi::Operand <MQ> opd_m_q;		\
		typedef zhetapi::Operand <MR> opd_m_r;		\
								\
		typedef zhetapi::Operand <MCZ> opd_m_cz;	\
		typedef zhetapi::Operand <MCQ> opd_m_cq;	\
		typedef zhetapi::Operand <MCR> opd_m_cr;	\
								\
		typedef zhetapi::Operand <B> opd_b;		\
		typedef zhetapi::Operand <S> opd_s;

#define	ONE	1

namespace zhetapi {

struct types {
	__TYPEDEFS__
	
	/////////////////////
	// Token factories
	/////////////////////

	static Token *one();

	template <class A>
	static Token *convert(A);
	
	/////////////////////
	// Token identifiers
	/////////////////////
	
	static bool is_zero(Token *);
	static bool is_one(Token *);

	static std::string symbol(const ::std::type_index &);
	static std::string proper_symbol(const ::std::type_index &);
};

}

#endif
