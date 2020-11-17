#ifndef NODE_H_
#define NODE_H_

// C/C++ headers
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>

// Engine headers
#include <token.hpp>

#include <hidden/label.hpp>
#include <hidden/class.hpp>

namespace zhetapi {

	struct node {
		// Members
		::std::shared_ptr <Token>	__tptr;
		lbl				__label;
		cls				__class;
		::std::vector <node>		__leaves;

		// Constructors
		node();
		node(Token *, const ::std::vector <node> &);
		node(Token *, lbl, const ::std::vector <node> &);
		
		// Binary
		node(Token *, const node &, const node &);
		
		// Member functions
		void transfer(const node &);

		// Printing
		void print(int = 1, int = 0) const;
		void print_no_address(int = 1, int = 0) const;

		::std::string display(int = 1, int = 0) const;

		// Static methods
		static bool loose_match(const node &, const node &);

		friend ::std::ostream &operator<<(::std::ostream &, const node &);
	};

}

#endif
