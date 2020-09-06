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
		 */
		enum type {
			opd,
			oph,
			opn,
			var,
			ftn
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

		// Any use?
		virtual token *copy() const = 0;

		// Any use?
		virtual bool operator==(token *) const = 0;
	};

	token::operator type() const
	{
		return caller();
	}
	
}

#endif
