#ifndef TOKEN_H_
#define TOKEN_H_

// C/C++ headers
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <exception>
#include <typeinfo>
#include <typeindex>
#include <mutex>

namespace zhetapi {

/** 
 * @brief Acts as a dummy class for
 * use of generic pointer in
 * other modules
 */
class Token {
public:
	// Maybe differentiate between const vector & and vector & methods
	using method = Token *(*)(Token *, const std::vector <Token *> &);
protected:
	// From ZHP (algorithm, function, etc)
	std::map <std::string, Token *>	_attributes;

	// From API (Token)
	std::map <std::string, method>	_methods;
public:
	Token();
	Token(const std::vector <std::pair <std::string, method>> &);

	virtual ~Token();

	Token *attr(const std::string &, const std::vector <Token *> &);
	
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
		token_node_list,
		token_module
	};

	bool operator!=(Token *) const;
	void list_attributes(std::ostream & = std::cout) const;

	// Change caller to a public member (static)

	/* 
	 * Inspector function passed on to all derived classes, helps to
	 * choose what to do with different Tokens from other classes.
	 */
	virtual type caller() const;
	virtual size_t id() const;

	/*
	 * Returns a representation of the Token, regardless of its
	 * type.
	 */
	virtual std::string dbg_str() const;

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

	class unknown_attribute {
		std::type_index _ti;
		std::string	_msg;
	public:
		unknown_attribute(const std::type_info &ti, const std::string &msg)
				: _ti(ti), _msg(msg) {}
		
		std::string what() const;
	};
};

// Token id macro
#define zhp_token_id(type)			\
	size_t type::id() const {		\
		return zhp_id <type> ();	\
	}

// Comparing tokens
bool tokcmp(Token *, Token *);

// Printing a list of tokens
std::ostream &operator<<(std::ostream &, const std::vector <Token *> &);
	
}

#endif
