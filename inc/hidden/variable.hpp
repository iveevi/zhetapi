#ifndef VARIALBE_H_
#define VARIALBE_H_

// C/C++ headers
#include <string>
#include <iostream>
#include <sstream>

// Engine headers
#include <token.hpp>
#include <types.hpp>

namespace zhetapi {

	template <class T, class U>
	class Variable {
		token	*__tptr;
	public:
		// Constructor
		Variable(token *);

		template <class A>
		Variable(A);

		// Destructor
		~Variable();

		// Reference
		const token *get() const;

		// Exceptions
		class illegal_type {};
	};

	// Constructors
	template <class T, class U>
	Variable <T, U> ::Variable(token *tptr) : __tptr(tptr) {}

	template <class T, class U>
	template <class A>
	Variable <T, U> ::Variable(A x)
	{
		__tptr = types <T, U> ::convert(x);

		if (!__tptr)
			throw illegal_type();
	}

	// Destructors
	template <class T, class U>
	Variable <T, U> ::~Variable()
	{
		delete __tptr;
	}

	// Reference
	template <class T, class U>
	const token *Variable <T, U> ::get() const
	{
		return __tptr;
	}
	
}

#endif
