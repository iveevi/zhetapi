#ifndef NODE_H_
#define NODE_H_

// C/C++ headers
#include <vector>
#include <memory>
#include <iostream>
#include <iterator>
#include <sstream>

// Engine headers
#include "../token.hpp"
#include "label.hpp"
#include "class.hpp"

namespace zhetapi {

class node {
public:
	using Leaves = std::vector <node>;
	using LeavesIt = Leaves::iterator;
	using LeavesCit = Leaves::const_iterator;
private:
	// Members
	Token *		_tptr		= nullptr;
	lbl		_label		= l_none;
	cls		_class		= c_none;
	size_t		_nodes		= 0;
	Leaves		_leaves;
public:
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

	// Operations for cleaner code
	template <class T>
	T *cast() const;

	Token *copy_token() const;

	node &operator[](size_t);
	const node &operator[](size_t) const;

	// Leaves modifiers
	LeavesIt begin();
	LeavesIt end();

	LeavesCit begin() const;
	LeavesCit end() const;

	LeavesIt insert(const LeavesIt &, const node &);
	LeavesIt erase(const LeavesIt &);

	void clear();

	// Properties
	lbl label() const;
	bool null() const;
	Token *ptr() const;
	bool empty() const;
	size_t child_count() const;
	Token::type caller() const;

	// Setters
	void relabel(lbl);
	void retokenize(Token *);
	void releaf(const Leaves &);

	void set_count(size_t);
	
	// Member functions
	void transfer(const node &);
	void copy_leaves(const node &);

	void append(const node &);
	void append_front(const node &);

	void remove_end();

	// Printing
	void print(int = 1, int = 0) const;
	void print_no_address(int = 1, int = 0) const;

	std::string display(int = 1, int = 0) const;

	// Static methods
	static bool loose_match(const node &, const node &);

	friend std::ostream &operator<<(std::ostream &, const node &);
};

template <class T>
T *node::cast() const
{
	return dynamic_cast <T *> (_tptr);
}

// Forward declarations
node factorize(const node &, const node &);

}

#endif
