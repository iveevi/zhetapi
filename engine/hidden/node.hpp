#ifndef NODE_H_
#define NODE_H_

// C/C++ headers
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>

// Engine headers
#include "../token.hpp"

#include "label.hpp"
#include "class.hpp"

namespace zhetapi {

	struct node {
		// Members
		std::shared_ptr <Token>	__tptr;
		lbl			__label;
		cls			__class;
		std::vector <node>	__leaves;

		// Constructors
		node();
		node(Token *, const std::vector <node> &);
		node(Token *, lbl, const std::vector <node> &);
		
		// Binary
		node(Token *, const node &, const node &);
		
		// Member functions
		void transfer(const node &);

		// Printing
		void print(int = 1, int = 0) const;
		void print_no_address(int = 1, int = 0) const;

		std::string display(int = 1, int = 0) const;

		// Static methods
		static bool loose_match(const node &, const node &);

		friend std::ostream &operator<<(std::ostream &, const node &);
	};

	node::node() : __tptr(nullptr), __label(l_none) {}
	
	node::node(Token *tptr, const std::vector <node> &leaves) :
			__leaves(leaves), __label(l_none)
	{
		__tptr.reset(tptr);
	}

	node::node(Token *tptr, lbl label, const std::vector <node> &leaves) :
		__leaves(leaves), __label(label)
	{
		__tptr.reset(tptr);
	}

	node::node(Token *tptr, const node &a, const node &b) : __leaves({a,
			b}), __label(l_none)
	{
		__tptr.reset(tptr);
	}

	void node::transfer(const node &ref)
	{
		__tptr = ref.__tptr;
		__label = ref.__label;
		__class = ref.__class;
		__leaves = ref.__leaves;
	}

	void node::print(int num, int lev) const
	{
		int counter = lev;
		while (counter > 0) {
			std::cout << "\t";
			counter--;
		}

		std::cout << "#" << num << ": " << __tptr->str() << " (" <<
			__tptr << ", " << strlabs[__label] << ") @ " << this <<
			std::endl;

		counter = 0;
		for (node itr : __leaves)
			itr.print(++counter, lev + 1);
	}

	void node::print_no_address(int num, int lev) const
	{
		int counter = lev;
		while (counter > 0) {
			std::cout << "\t";
			counter--;
		}

		std::cout << "#" << num << ": " << __tptr->str() << " (" <<
			strlabs[__label] << ")" << std::endl;

		counter = 0;
		for (node itr : __leaves)
			itr.print_no_address(++counter, lev + 1);
	}
	
	std::string node::display(int num, int lev) const
	{
		std::ostringstream oss;

		oss << std::endl;

		int counter = lev;
		while (counter > 0) {
			oss << "\t";
			counter--;
		}

		oss << "#" << num << ": " << __tptr->str() << " (" << __tptr << ") @ "
			<< this << std::endl;

		counter = 0;
		for (node itr : __leaves)
			oss << itr.display(++counter, lev + 1);

		return oss.str();
	}

	bool node::loose_match(const node &a, const node &b)
	{
		// Check the Token
		if (*(a.__tptr.get()) != b.__tptr.get())
			return false;
		
		// Check the leaves
		if (a.__leaves.size() != b.__leaves.size())
			return false;

		for (size_t i = 0; i < a.__leaves.size(); i++) {
			if (!loose_match(a.__leaves[i], b.__leaves[i]))
				return false;
		}

		return true;
	}
	
	std::ostream &operator<<(std::ostream &os, const node &tree)
	{
		os << tree.display();

		return os;
	}

}

#endif
