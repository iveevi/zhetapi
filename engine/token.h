#ifndef TOKEN_H_
#define TOKEN_H_

// C++ Standard Libraries
#include <string>

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
	enum type {
		OPERAND,
		OPERATION,
		VARIABLE,
		FUNCTOR
	};

	/* Virtual:
	 * [type] [caller]() - inspector function passed
	 * on to all derived classes */
	virtual type caller() const = 0;

	/* Virtual:
	 * string [str]() - returns the string
	 * representation of the token */
	virtual std::string str() const = 0;

	virtual token *copy() const = 0;

	virtual bool operator==(token *) const = 0;
};

#endif
