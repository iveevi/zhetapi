#ifndef PARAMETRIZATION_H_
#define PARAMETRIZATION_H_

// C++ headers
#include <cstdlib>

// Engine headers
#include "vector.hpp"

namespace zhetapi {

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
	Vector <T> dpos() const;	// Derivative of vector position
	bool step();			// Step through the parametrization

	void reset();			// Reset the position

	static const size_t partition_size;
};

// Static variables
template <class T, class P, class F>
const size_t Parametrization <T, P, F> ::partition_size = 1000;

// Constructors
template <class T, class P, class F>
Parametrization <T, P, F> ::Parametrization(F ftn, P par, T start, T end)
		: _ftn(ftn), _par(par),
		_start(start), _end(end),
		_pos(start)
{
	_inc = (end - start) / (T) partition_size;
}

template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::value() const
{
	return _ftn(_par(_pos));
}

template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::pos() const
{
	return _par(_pos);
}

template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::dpos() const
{
	return (_par(_pos + _inc) - _par(_pos))/_inc;
}

template <class T, class P, class F>
bool Parametrization <T, P, F> ::step()
{
	_pos += _inc;

	// Cycle back to the start;
	if (_pos > _end) {
		_pos = _start;

		return true;
	}
	
	return false;
}

template <class T, class P, class F>
void Parametrization <T, P, F> ::reset()
{
	_pos = _start;
}

}

#endif
