#ifndef TYPES_H_
#define TYPES_H_

// Engine headers
#include <operand.hpp>
#include <rational.hpp>
#include <complex.hpp>
#include <vector.hpp>
#include <matrix.hpp>

#define __TYPEDEFS__				\
		typedef U Z;			\
		typedef Rational <Z> Q;		\
		typedef T R;			\
						\
		typedef Complex <R> CR;         \
		typedef Complex <Q> CQ;         \
						\
		typedef Vector <R> VR;          \
		typedef Vector <Q> VQ;          \
						\
		typedef Vector <CR> VCR;        \
		typedef Vector <CQ> VCQ;        \
						\
		typedef Matrix <R> MR;          \
		typedef Matrix <Q> MQ;          \
						\
		typedef Matrix <CR> MCR;        \
		typedef Matrix <CQ> MCQ;        \
						\
		typedef operand <Z> opd_z;      \
		typedef operand <R> opd_r;      \
		typedef operand <Q> opd_q;      \
						\
		typedef operand <CR> opd_cr;	\
		typedef operand <CQ> opd_cq;	\
						\
		typedef operand <VR> opd_v_r;	\
		typedef operand <VQ> opd_v_q;	\
						\
		typedef operand <VCR> opd_v_cr;	\
		typedef operand <VCQ> opd_v_cq;	\
						\
		typedef operand <MR> opd_m_r;	\
		typedef operand <MQ> opd_m_q;	\
						\
		typedef operand <MCR> opd_m_cr;	\
		typedef operand <MCQ> opd_m_cq;

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