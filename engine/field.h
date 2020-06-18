#ifndef FIELD_H_
#define FIELD_H_

#include "config.h"

/**
 * @brief Contains a set of operations
 * applicable to a set/field of elements
 * which are of type/set T.
 */
template <class T>
struct field {
	config <T> cfg;
	vtable <T> vtbl;

	field();
	field(const config <T> &, const vtable <T> &);
};

template <class T>
field <T> ::field() : cfg(), vtbl() {}

// Specialization for defaults go here

template <class T>
field <T> ::field(const config <T> &c, const vtable <T> &vt)
	: cfg(c), vtbl() {}

#endif
