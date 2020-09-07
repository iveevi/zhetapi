#ifndef NODE_MANAGER_H_
#define NODE_MADAGER_H_

// Engine headers
#include <parser.hpp>
#include <barn.hpp>
#include <types.hpp>

namespace zhetapi {

	template <class T, class U>
	class node_manager {
		Barn <T, U>			__barn;
		node				__tree;
		std::vector <std::string>	__params;
	public:
		node_manager(const std::string &, Barn <T, U> = Barn <T, U> ());

		token *value() const;

		/*
		 * Responsible for expanding varialbe clusters and truning them
		 * into product of operands.
		 */
		void expand(node &);

		void print() const;
	private:
		token *value(node) const;
		
		void label(node &) const;

		node expand(const std::string &);

		/*
		 * Node factories; produce special nodes such as ones, zeros,
		 * etc. to make constuction of such nodes easy.
		 */
		static node nf_one();
		static node nf_zero();
	public:
		// General error
		class error {
			std::string str;
		public:
			error(std::string s) : str(s) {}

			const std::string &what() const {
				return str;
			}
		};

		// Syntax error
		class syntax_error : public error {
		public:
			syntax_error(std::string s) : error(s) {}
		};

		// Undefined symbol error
		class undefined_symbol : public error {
		public:
			undefined_symbol(std::string s) : error(s) {}
		};
	};

	template <class T, class U>
	node_manager <T, U> ::node_manager(const std::string &str, Barn <T, U>
			barn) : __barn(barn) 
	{
		zhetapi::parser <T, U> pr(__barn);

		siter iter = str.begin();
		siter end = str.end();

		bool r = qi::phrase_parse(iter, end, pr, qi::space, __tree);

		// Unpack variable clusters
		expand(__tree);

		// Label the tree
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
		
		token *tptr;

		switch (*(tree.__tptr)) {
		case token::opd:
			return tree.__tptr.get();
		case token::oph:
			for (node leaf : tree.__leaves) {
				token *tptr = value(leaf);

				types.push_back(typeid(*tptr));
				values.push_back(tptr);
			}

			tptr = __barn.value((dynamic_cast <operation_holder *>
						(tree.__tptr.get()))->rep, types,
					values);

			return tptr;
		}

		return nullptr;
	}

	// Expansion methods
	template <class T, class U>
	void node_manager <T, U> ::expand(node &ref)
	{
		if (*(ref.__tptr) == token::vcl) {
			/*
			 * Excluding the parameters, the variable cluster should
			 * always be a leaf of the tree.
			 */
			assert(ref.__leaves.size() == 0);

			variable_cluster *vclptr = dynamic_cast
				<variable_cluster *> (ref.__tptr.get());

			ref = expand(vclptr->__cluster);
		}

		for (node &leaf : ref.__leaves)
			expand(leaf);
	}

	template <class T, class U>
	node node_manager <T, U> ::expand(const std::string &str)
	{
		node out = nf_one();

		std::string tmp;

		for (size_t i = 0; i < str.length(); i++) {
			tmp += str[i];

			token *tptr = __barn.get(tmp);

			if (tptr != nullptr) {
				out = node(new operation_holder("*"), out,
						node(tptr, {}));

				tmp.clear();
			}
		}

		/*
		 * If tmp is not empty, it implies that we could not find a
		 * match for it, and therefore the parsing is incomplete.
		 */
		if (!tmp.empty())
			throw undefined_symbol(tmp);

		return out;
	}

	template <class T, class U>
	void node_manager <T, U> ::label(node &ref) const
	{
		switch (ref.__tptr->caller()) {
		case token::opd:
			ref.__label = constant_label <T, U> (ref.__tptr.get());
			break;
		case token::oph:
			ref.__label = l_operation_constant;

			for (node &leaf : ref.__leaves)
				label(leaf);

			break;
		}
	}

	// Node factories
	template <class T, class U>
	node node_manager <T, U> ::nf_one()
	{
		return node(new operand <U> (1), {});
	}

}

#endif
