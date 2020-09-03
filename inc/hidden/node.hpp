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
		
		// Member functions

		// Printing
		void print(int = 1, int = 0) const;

		std::string display(int = 1, int = 0) const;

		friend std::ostream &operator<<(std::ostream &, const node &);
	};

	node::node() : __tptr(nullptr), __label(l_none) {}
	
	node::node(token *tptr, const std::vector <node> &leaves) :
		__leaves(leaves), __label(l_none)
	{
		__tptr.reset(tptr);
	}

	void node::print(int num, int lev) const
	{
		int counter = lev;
		while (counter > 0) {
			std::cout << "\t";
			counter--;
		}

		std::cout << "#" << num << " - " << __tptr->str() << " (" <<
			__tptr << ", " << strlabs[__label] << ") @ " << this <<
			std::endl;

		counter = 0;
		for (node itr : __leaves)
			itr.print(++counter, lev + 1);
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

		oss << "#" << num << " - " << __tptr->str() << " (" << __tptr << ") @ "
			<< this << std::endl;

		counter = 0;
		for (node itr : __leaves)
			oss << itr.display(++counter, lev + 1);

		return oss.str();
	}
	
	std::ostream &operator<<(std::ostream &os, const node &tree)
	{
		os << tree.display();
	}

}

#endif
