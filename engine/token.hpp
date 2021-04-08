#ifndef TOKEN_H_
#define TOKEN_H_

// C/C++ headers
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace zhetapi {

/** 
 * @brief Acts as a dummy class for
 * use of generic pointer in
 * other modules
 */
class Token {
public:
	// Maybe differentiate between const vector & and vector & methods
	using method = Token *(*)(const std::vector <Token *> &);
protected:
	std::map <std::string, Token *>	_attributes;
	std::map <std::string, method>	_methods;
public:
	// Token()
	virtual ~Token();
	
	/*
	 * Codes used to identify the Token, more on the is presented
	 * below. Should not be used by the user.
	 *
	 * Codes:
	 * 	opd - Operand
	 * 	oph - operation (place holder)
	 * 	opn - operation
	 * 	var - variable
	 * 	vrh - variable (place holder)
	 * 	vcl - variable cluster (place holder)
	 * 	ftn - function
	 * 	ndr - node reference
	 *	ndd - node differential
	 *	reg - engine registrable
	 * 	wld - wildcard
	 */
	enum type {
		undefined,
		alg,
		opd,
		oph,
		opn,
		var,
		vrh,
		vcl,
		ftn,
		ndr,
		ndd,
		reg,
		token_wildcard,
		token_lvalue,
		token_rvalue,
		token_node_list
	};

	bool operator!=(Token *) const;

	// Change caller to a public member (static)

	/* 
	 * Inspector function passed on to all derived classes, helps to
	 * choose what to do with different Tokens from other classes.
	 */
	virtual type caller() const = 0;

	/*
	 * Returns a representation of the Token, regardless of its
	 * type.
	 */
	virtual std::string str() const = 0;

	// Add a virtual display method

	/*
	 * Returns a heap allocated copy of the Token. Used in copy
	 * constructors for nodes and engines.
	 */
	virtual Token *copy() const = 0;

	/*
	 * Compares Tokens and returns their similarity. Used for node
	 * matching.
	 */
	virtual bool operator==(Token *) const = 0;

	// virtual Token *value()?
};

// Comparing tokens
bool tokcmp(Token *, Token *);

// Printing a list of tokens
std::ostream &operator<<(std::ostream &, const std::vector <Token *> &);
	
}

#endif
