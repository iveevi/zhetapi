#ifndef PARAMETRIZATION_H_
#define PARAMETRIZATION_H_

// C++ headers
#include <cstdlib>

// Engine headers
#include <vector.hpp>

template <class T, class P, class F>
class Parametrization {
	T	__start;
	T	__end;

	T	__pos;			// Current position
	T	__inc;			// Increment

	P	__par;			// Single variable vector function
	F	__ftn;			// Multivariable vector function
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
		: __ftn(ftn), __par(par),
		__start(start), __end(end)
{
	__pos = start;

	__inc = (end - start) / (T) partition_size;
}

template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::value() const
{
	return __ftn(__par(__pos));
}

template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::pos() const
{
	return __par(__pos);
}

template <class T, class P, class F>
Vector <T> Parametrization <T, P, F> ::dpos() const
{
	return (__par(__pos + __inc) - __par(__pos))/__inc;
}

template <class T, class P, class F>
bool Parametrization <T, P, F> ::step()
{
	__pos += __inc;

	// Cycle back to the start;
	if (__pos > __end) {
		__pos = __start;

		return true;
	}
	
	return false;
}

template <class T, class P, class F>
void Parametrization <T, P, F> ::reset()
{
	__pos = __start;
}

#endif