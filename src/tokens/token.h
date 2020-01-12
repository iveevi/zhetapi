#ifndef TYPES_H
#define TYPES_H

#include <cctype>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <sstream>
#include <cmath>

#include "../misc/debug.h"
// #include "trees.h"

namespace tokens {
	/* Default data/info type to
	 * use for calculations */
	typedef double def_t;

	/* Token Class:
	 *
	 * Acts as a dummy class for
	 * use of generic pointer in
	 * other modules */
	class token {
	public:
		/* Enumerations:
		 * [type] - new data type to allow function
		 * caller inspection */
		enum type {NONE, OPERAND, OPERATION,
			VARIABLE, FUNCTION, MODULE};

		/* Virtual:
		 * [type] [caller]() - inspector function passed
		 * on to all derived classes */
		virtual type caller();

                /* Virtual:
		 * string [str]() - returns the string
		 * representation of the token */
		virtual std::string str() const;
	};

	token::type token::caller()
	{
		return NONE;
	}

        std::string token::str() const
        {
                return "NA";
        }
}

#endif
