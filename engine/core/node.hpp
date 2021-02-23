#ifndef NODE_H_
#define NODE_H_

// C/C++ headers
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>

// Engine headers
#include <token.hpp>

#include <core/label.hpp>
#include <core/class.hpp>

namespace zhetapi {

struct node {
	// Members
	std::shared_ptr <Token>	__tptr		= nullptr;
	lbl			__label		= l_none;
	cls			__class		= c_none;
	std::vector <node>	__leaves	= {};
	size_t			__nodes		= 0;

	// Constructors
	node();
	node(Token *, const ::std::vector <node> &);
	node(Token *, lbl, const ::std::vector <node> &);

	// Unary
	explicit node(Token *, const node &, bool);
	
	// Binary
	explicit node(Token *, const node &, const node &);

	// Properties
	bool empty() const;
	
	// Member functions
	void transfer(const node &);
	void append(const node &);

	// Printing
	void print(int = 1, int = 0) const;
	void print_no_address(int = 1, int = 0) const;

	std::string display(int = 1, int = 0) const;

	// Static methods
	static bool loose_match(const node &, const node &);

	friend ::std::ostream &operator<<(::std::ostream &, const node &);
};

}

#endif
