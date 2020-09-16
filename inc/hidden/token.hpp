#ifndef TOKEN_H_
#define TOKEN_H_

// C/C++ headers
#include <string>

namespace zhetapi {

	/** 
	 * @brief Acts as a dummy class for
	 * use of generic pointer in
	 * other modules
	 */
	class token {
	public:
		/*
		 * Codes used to identify the token, more on the is presented
		 * below. Should not be used by the user.
		 *
		 * Codes:
		 * 	opd - operand
		 * 	oph - operation (place holder)
		 * 	opn - operation
		 * 	var - variable
		 * 	vrh - variable (place holder)
		 * 	vcl - variable cluster (place holder)
		 * 	ftn - function
		 * 	ndr - node reference
		 * 	wld - wildcard
		 */
		enum type {
			opd,
			oph,
			opn,
			var,
			vrh,
			vcl,
			ftn,
			ndr,
			wld
		};

		/*
		 * Implicit conversion operator for cleaner notation, uses the
		 * inspector function defined below.
		 */
		operator type() const;

		/* 
		 * Inspector function passed on to all derived classes, helps to
		 * choose what to do with different tokens from other classes.
		 */
		virtual type caller() const = 0;

		/*
		 * Returns a representation of the token, regardless of its
		 * type.
		 */
		virtual std::string str() const = 0;

		/*
		 * Returns a heap allocated copy of the token. Used in copy
		 * constructors for nodes and barns.
		 */
		virtual token *copy() const = 0;

		/*
		 * Compares tokens and returns their similarity. Used for node
		 * matching.
		 */
		virtual bool operator==(token *) const = 0;
	};

	token::operator type() const
	{
		return caller();
	}
	
}

#endif
