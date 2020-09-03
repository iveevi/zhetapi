#ifndef TYPES_H_
#define TYPES_H_

// Engine headers
#include <operand.hpp>
#include <rational.hpp>
#include <complex.hpp>
#include <vector.hpp>
#include <matrix.hpp>

#define __TYPEDEFS__						\
		typedef U Z;					\
		typedef Rational <Z> Q;				\
		typedef T R;					\
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
		typedef zhetapi::operand <Z> opd_z;      	\
		typedef zhetapi::operand <Q> opd_q;      	\
		typedef zhetapi::operand <R> opd_r;      	\
								\
		typedef zhetapi::operand <CZ> opd_cz;		\
		typedef zhetapi::operand <CQ> opd_cq;		\
		typedef zhetapi::operand <CR> opd_cr;		\
								\
		typedef zhetapi::operand <VZ> opd_v_z;		\
		typedef zhetapi::operand <VQ> opd_v_q;		\
		typedef zhetapi::operand <VR> opd_v_r;		\
								\
		typedef zhetapi::operand <VCZ> opd_v_cz;	\
		typedef zhetapi::operand <VCQ> opd_v_cq;	\
		typedef zhetapi::operand <VCR> opd_v_cr;	\
								\
		typedef zhetapi::operand <MZ> opd_m_z;		\
		typedef zhetapi::operand <MQ> opd_m_q;		\
		typedef zhetapi::operand <MR> opd_m_r;		\
								\
		typedef zhetapi::operand <MCZ> opd_m_cz;	\
		typedef zhetapi::operand <MCQ> opd_m_cq;	\
		typedef zhetapi::operand <MCR> opd_m_cr;

#define	ONE	1

namespace zhetapi {

	/*
	 * TYPES:
	 *
	 * Z - Integer
	 * Q - Rational
	 * R - Real
	 *
	 */

	template <class T, class U>
	class types {
	public:
		__TYPEDEFS__
		
		/////////////////////
		// Token factories
		/////////////////////

		static token *one(); 
		
		/////////////////////
		// Token identifiers
		/////////////////////
		
		static bool is_zero(token *tptr);
		static bool is_one(token *tptr);
	};

	template <class T, class U>
	token *types <T, U> ::one()
	{
		return new opd_z(U (ONE));
	}

	template <class T, class U>
	bool types <T, U> ::is_zero(token *tptr)
	{
		opd_z *_oz = dynamic_cast <opd_z *> (tptr);
		if (_oz != nullptr)
			return (_oz->get() == 0);
		
		opd_q *_oq = dynamic_cast <opd_q *> (tptr);
		if (_oq != nullptr)
			return (_oq->get() == 0);
		
		opd_r *_or = dynamic_cast <opd_r *> (tptr);
		if (_or != nullptr)
			return (_or->get() == 0);

		return false;
	}

	template <class T, class U>
	bool types <T, U> ::is_one(token *tptr)
	{
		opd_z *_oz = dynamic_cast <opd_z *> (tptr);
		if (_oz != nullptr)
			return (_oz->get() == 1);
		
		opd_q *_oq = dynamic_cast <opd_q *> (tptr);
		if (_oq != nullptr)
			return (_oq->get() == 1);
		
		opd_r *_or = dynamic_cast <opd_r *> (tptr);
		if (_or != nullptr)
			return (_or->get() == 1);

		return false;
	}

}

#endif
