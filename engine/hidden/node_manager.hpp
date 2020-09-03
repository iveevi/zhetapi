#ifndef NODE_MANAGER_H_
#define NODE_MADAGER_H_

// Engine headers
#include <parser.hpp>
// #include <barn.hpp>

namespace zhetapi {

	template <class T, class U>
	class node_manager {
		// Barn <T, U>			__brn;
		node				__tree;
		std::vector <std::string>	__params;
	public:
		node_manager(const std::string &);

		void print() const;
	private:
		void label(node &) const;
	};

	template <class T, class U>
	node_manager <T, U> ::node_manager(const std::string &str)
	{
		zhetapi::parser <T, U> pr;

		siter iter = str.begin();
		siter end = str.end();

		bool r = qi::parse(iter, end, pr, __tree);

		label(__tree);
	
		std::cout << "-------------------------\nstr: " << str <<
			std::endl;

		if (r) {
			// Status
			std::cout << "Parsing succeeded";

			if (iter != end)
				std::cout << " (NOT FULLY PARSED)";

			std::cout << std::endl;

			// Node
			std::cout << "nd:" << std::endl;
			__tree.print();
		} else {
			std::cout << "Parsing failed" << std::endl;
		}
	}

	template <class T, class U>
	void node_manager <T, U> ::label(node &ref) const
	{
		switch (ref.__tptr->caller()) {
		case token::OPERAND:
			ref.__label = constant_label <T, U> (ref.__tptr.get());
			break;
		}
	}

}

#endif
