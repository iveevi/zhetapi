#ifndef TYPES_H_
#define TYPES_H_

// Engine headers
#include <operand.h>
#include <rational.h>
#include <complex.h>
#include <vector.h>
#include <matrix.h>

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
	/////////////////////
	// Aliases
	/////////////////////
	
	// Raw types
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

	// Wrapped types
	using opd_z = operand <Z>;
	using opd_r = operand <R>;
	using opd_q = operand <Q>;

	using opd_cr = operand <CR>;
	using opd_cq = operand <CQ>;
	
	/////////////////////
	// Token identifiers
	/////////////////////
	
	static is_zero(token *tptr);
	static is_one(token *tptr);
};

template <class T, class U>
bool types <T, U> ::is_zero(token *tptr)
{
	opd_z *_oz = dynamic_cast <opd_z *> (tptr);
	if (_oz != nullptr)
		return (_oz->get() == 0);
	
	opd_q *_oq = dynamic_cast <opd_q *> (tptr);
	if (_oq != nullptr)
		return (_oq->get() == 0);
	
	opd_z *_or = dynamic_cast <opd_r *> (tptr);
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
	
	opd_z *_or = dynamic_cast <opd_r *> (tptr);
	if (_or != nullptr)
		return (_or->get() == 1);

	return false;
}

#endif
