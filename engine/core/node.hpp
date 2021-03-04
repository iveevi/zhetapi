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
	Token *			__tptr		= nullptr;
	lbl			__label		= l_none;
	cls			__class		= c_none;
	size_t			__nodes		= 0;
	std::vector <node>	__leaves	= {};

	// Constructors
	node();
	node(const node &);
	
	node(Token *);
	node(Token *, lbl);
	node(Token *, const node &, bool);
	node(Token *, const node &, const node &);
	node(Token *, const std::vector <node> &);
	node(Token *, lbl, const std::vector <node> &);

	node &operator=(const node &);

	~node();

	// Properties
	bool empty() const;
	
	// Member functions
	void transfer(const node &);

	void append(const node &);
	void append_front(const node &);

	// Printing
	void print(int = 1, int = 0) const;
	void print_no_address(int = 1, int = 0) const;

	std::string display(int = 1, int = 0) const;

	// Static methods
	static bool loose_match(const node &, const node &);

	friend std::ostream &operator<<(std::ostream &, const node &);
};

}

#endif
