#ifndef NODE_H_
#define NODE_H_

// C/C++ headers
#include <vector>
#include <memory>

// Engine headers
#include <token.hpp>
#include <label.hpp>
#include <class.hpp>

namespace zhetapi {

	struct node {
		// Members
		std::shared_ptr <token>	__tptr;
		lbl			__label;
		cls			__class;
		std::vector <node>	__leaves;

		// Constructors and deconstructors
		node();
		node(token *, const std::vector <node> &);
		
		~node();

		// Member functions
		void print(int = 1, int = 0) const;
	};

	node::node() : __tptr(nullptr) {}
	
	node::node(token *tptr, const std::vector <node> &leaves) :
		__leaves(leaves)
	{
		__tptr.reset(tptr);

		std::cout << "tok: " << __tptr << " -> " << __tptr->str() << std::endl;
	}

	void node::print(int num, int lev) const
	{
		int counter = lev;
		while (counter > 0) {
			std::cout << "\t";
			counter--;
		}

		std::cout << "#" << num << " - " << __tptr->str() << " (" << __tptr << ") @ "
			<< this << std::endl;

		counter = 0;
		for (node itr : __leaves)
			itr.print(++counter, lev + 1);
	}
}

#endif
