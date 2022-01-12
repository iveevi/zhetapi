#ifndef PARAMETRIZATION_H_
#define PARAMETRIZATION_H_

// C++ headers
#include <cstdlib>

// Engine headers
#include "vector.hpp"

namespace zhetapi {

/**
 * @brief Represents the parametrization of a curve. With parameters T, P and F,
 * the value of the parametrization at any value t (of type T) is \f$F(P(t))\f$
 *
 * Mathematically, given that T is a set of some quantity representing a scalar,
 * then
 *
 * \f$P: T \to T^m\f$
 *
 * and
 *
 * \f$F: T^m \to T^n.\f$
 *
 * The parametrization function is thus the map \f$T \to T^m.\f$
 *
 * Terminology:
 *
 * - The \b time of the parametrization is the value \f$t\f$ that ranges from
 *   the starting and ending T values.
 * - The \b position of the parametrization, \f$P(t)\f$ is the value of the
 *   secondary transformation at a current time.
 * - The \b value of the parametrization, \f$F(P(t))\f$ is the value of the
 *   primary transformation at the value of the secondary transformation at the
 *   current time.
 *
 * @tparam T the basic operating type.
 * @tparam P the type of the single variable vector function (the secondary
 * transformation).
 * @tparam F the type of the multivariable vector function (the primary
 * transformation).
 */
template <class T, class P, class F>
class Parametrization {
	T	_start;
	T	_end;

	T	_pos;			// Current position
	T	_inc;			// Increment

	P	_par;			// Single variable vector function
	F	_ftn;			// Multivariable vector function
public:
	Parametrization(F, P, T, T);

	Vector <T> value() const;	// Value at position
	Vector <T> pos() const;		// Vector position
	T time() const;

	Vector <T> dpos() const;	// Derivative of vector position
	
	bool step();			// Step through the parametrization

	void reset();			// Reset the position

	static const size_t partition_size;
};

/**
 * @brief The default number of steps that a parametrization will undergo.
 * Default value is 1000.
 */
template <class T, class P, class F>
const size_t Parametrization <T, P, F> ::partition_size = 1000;

/**
 * @brief Constructs a parametrization with given primary and secondary
 * transformations, and the bounds of the parametrization. The state of the
 * parametrization is set to the start.
 *
 * @param ftn the primary transformation.
 * @param par the secondary transformation.
 * @param start the beginning value of the parametrization.
 * @param end the ending value of the parametrization.
 */
template <class T, class P, class F>
Parametrization <T, P, F> ::Parametrization(F ftn, P par, T start, T end)
		: _ftn(ftn), _par(par),
		_start(start), _end(end),
		_pos(start)
{
	_inc = (end - start) / (T) partition_size;
}

/**
 * @return the value of the parametrization at the current time.
 */
template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::value() const
{
	return _ftn(_par(_pos));
}

/**
 * @return the position of the parametrization at the current time.
 */
template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::pos() const
{
	return _par(_pos);
}

/**
 * @return the current time.
 */
template <class T, class P, class F>
T Parametrization <T, P, F> ::time() const
{
	return _pos;
}

/**
 * @brief The derivative of the secondary transformation, evaluated using a
 * forward difference.
 *
 * @return the derivative of the secondary transformation at the current time.
 */
template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::dpos() const
{
	return (_par(_pos + _inc) - _par(_pos))/_inc;
}

/**
 * @brief Increments the current time of the parametrization according to the
 * static variable \p partition_size.
 *
 * @return \c true if the parametrization has reached the end of the its bounds
 * and \c false otherwise.
 */
template <class T, class P, class F>
bool Parametrization <T, P, F> ::step()
{
	// TODO: allow multiple steps at once
	_pos += _inc;

	// Cycle back to the start;
	if (_pos > _end) {
		_pos = _start;

		return true;
	}

	return false;
}

/**
 * @brief Resets the current time of the parametrization to the starting time.
 */
template <class T, class P, class F>
void Parametrization <T, P, F> ::reset()
{
	_pos = _start;
}

}

#endif
