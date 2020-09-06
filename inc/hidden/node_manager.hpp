#ifndef NODE_MANAGER_H_
#define NODE_MADAGER_H_

// Engine headers
#include <parser.hpp>
#include <barn.hpp>

namespace zhetapi {

	template <class T, class U>
	class node_manager {
		Barn <T, U>			__brn;
		node				__tree;
		std::vector <std::string>	__params;
	public:
		node_manager(const std::string &);

		token *value() const;

		void print() const;
	private:
		token *value(node) const;
		
		void label(node &) const;
	};

	template <class T, class U>
	node_manager <T, U> ::node_manager(const std::string &str)
	{
		zhetapi::parser <T, U> pr;

		siter iter = str.begin();
		siter end = str.end();

		bool r = qi::phrase_parse(iter, end, pr, qi::space, __tree);

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

	// Value finding methods
	template <class T, class U>
	token *node_manager <T, U> ::value() const
	{
		return value(__tree);
	}

	template <class T, class U>
	token *node_manager <T, U> ::value(node tree) const
	{
		std::vector <token *> values;

		std::vector <std::type_index> types;
		
		using namespace std;

		token *tptr;

		switch (tree.__tptr->caller()) {
		case token::OPERAND:
			return tree.__tptr.get();
		case token::OPERATION_HOLDER:
			for (node leaf : tree.__leaves) {
				token *tptr = value(leaf);

				types.push_back(typeid(*tptr));
				values.push_back(tptr);
			}

			tptr = __brn.value((dynamic_cast <operation_holder *>
						(tree.__tptr.get()))->rep, types,
					values);

			return tptr;
		}

		return nullptr;
	}

	template <class T, class U>
	void node_manager <T, U> ::label(node &ref) const
	{
		switch (ref.__tptr->caller()) {
		case token::OPERAND:
			ref.__label = constant_label <T, U> (ref.__tptr.get());
			break;
		case token::OPERATION_HOLDER:
			ref.__label = l_operation_constant;

			for (node &leaf : ref.__leaves)
				label(leaf);

			break;
		}
	}

}

#endif
