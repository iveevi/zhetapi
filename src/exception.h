#ifndef EXCEPTION_H
#define EXCEPTION_H

// C++ Standard Libraries
#include <string>

// Custom Built Libraries
#include "token.h"

namespace tokens {
	/* Exception Class:
	 *
	 * Represents the generic error that
	 * is thrown whenever an error occurs
	 * when trying to compute with the
	 * operands */
	class exception {
	protected:
		/* std::string [msg] - what the
		 * error is about */
		std::string msg;
	public:
		/* Constructors:
		 * exception() - default constructor
		 *   that sets the private member [msg]
		 *   to an empty string
		 * exception(std::string) - sets the
		 *   private member variable [msg] to
		 *   whatever string is passed */
		exception();
		explicit exception(std::string);

		/* Virtualized Member Functions:
		 * void [set](std::string) - sets the
		 *   private member variable [msg] to
		 *   whatever string is passed
		 * std::string [what]() - returns a constant
		 *   (unchangeable) reference to the contents
		 *   of the private member variable [msg] */
		virtual void set(std::string);
		
		virtual const std::string &what();
	};

	/* Exception Class Member Functions
	 *
	 * See class declaration for a
	 * description of each function
	 *
	 * Constructors: */
	exception::exception() : msg("") {}
	exception::exception(std::string str) : msg(std::move(str)) {}

	/* Virtualized member functions:
	 * setter and getters */
	void exception::set(std::string str)
	{
		msg = std::move(str);
	}

	const std::string &exception::what()
	{
		return msg;
	}
}

#endif