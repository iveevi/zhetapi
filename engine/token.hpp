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

// Forward declarations
class Engine;
class MethodTable;

/** 
 * @brief The basic unit of computation for the ZHP scripting language and
 * framework.
 */
class Token {
	MethodTable *_mtable = nullptr;
public:
	/*
	 * TODO: make this redundant
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
		token_module,
		token_collection,
		token_dictionary
	};
public:
	Token();
	Token(MethodTable *);
	// Token(const std::vector <std::pair <std::string, method>> &);

	virtual ~Token();

	// TODO: also need to add documentation for methods
	virtual Token *attr(const std::string &, Engine *,
			const std::vector <Token *> &, size_t);
	virtual void list_attributes(std::ostream & = std::cout) const;

	bool operator!=(Token *) const;

	// Change caller to a public member (static)

	/* 
	 * Inspector function passed on to all derived classes, helps to
	 * choose what to do with different Tokens from other classes.
	 */
	virtual type caller() const;
	virtual uint8_t id() const;

	/*
	 * Returns a representation of the Token, regardless of its
	 * type.
	 */
	virtual std::string dbg_str() const;

	// TODO: Add a virtual display method

	/*
	 * Compares Tokens and returns their similarity. Used for node
	 * matching.
	 */
	virtual bool operator==(Token *) const;

	// Read and write
	virtual void write(std::ostream &) const;

	/**
	 * @brief Returns a copy of the Token (with the same data: the resulting
	 * Token should equal the original with ==). Pure virtual because any
	 * Tokens used will be copied at some point.
	 */
	virtual Token *copy() const = 0;

	/**
	 * @brief Thrown if the program requests a Token for an attribute or
	 * method it does not have.
	 */
	class unknown_attribute : public std::runtime_error {
	public:
		unknown_attribute(const std::string &msg)
				: std::runtime_error(msg) {}
	};

	/**
	 * @brief Thrown if the Token does not have a write function.
	 */
	class empty_io : public std::runtime_error {
	public:
		empty_io() : std::runtime_error("Empty IO functions (write)...") {}
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

// Macro for defining methods
#define TOKEN_METHOD(name)	\
	Token *name(Token *tptr, const Targs &args)

}

#endif
