#ifndef NODE_MANAGER_H_
#define NODE_MANAGER_H_

// Engine headers
#include <barn.hpp>
#include <node_reference.hpp>
#include <parser.hpp>
#include <types.hpp>
#include <variable_holder.hpp>

namespace zhetapi {

	template <class T, class U>
	class node_manager {
		/*
		 * Internal (common between nodes) barn class used to make
		 * decisions about computation.
		 */
		Barn <T, U>			__barn;

		/*
		 * The internal representation of the function or expression.
		 */
		node				__tree;

		/*
		 * List of parameters in node form. References are made to these
		 * nodes whenever a variable node is encoutered.
		 */
		std::vector <node>		__refs;

		/*
		 * List of paramaters in string form (by their symbols).
		 */
		std::vector <std::string>	__params;
	public:
		node_manager();
		node_manager(const std::string &, Barn <T, U> = Barn <T, U> ());
		node_manager(const std::string &, const std::vector
				<std::string> &, Barn <T, U> = Barn <T, U> ());

		node_manager(const node_manager &);

		node_manager &operator=(const node_manager &);

		token *value() const;

		token *substitute_and_compute(std::vector <token *> &);

		/*
		 * Responsible for expanding varialbe clusters and truning them
		 * into product of operands.
		 */
		void expand(node &);

		void print() const;
	private:
		token *value(node) const;
		
		void label(node &) const;
		
		void rereference(node &);

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
	node_manager <T, U> ::node_manager() {}

	template <class T, class U>
	node_manager <T, U> ::node_manager(const std::string &str, Barn <T, U>
			barn) : __barn(barn) 
	{
		zhetapi::parser <T, U> pr;

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
		
		std::cout << "-------------------------\nstr: " << str <<
			std::endl;
	}

	template <class T, class U>
	node_manager <T, U> ::node_manager(const std::string &str, const
			std::vector <std::string> &params, Barn <T, U> barn) :
		__params(params), __barn(barn) 
	{
		zhetapi::parser <T, U> pr;

		siter iter = str.begin();
		siter end = str.end();

		bool r = qi::phrase_parse(iter, end, pr, qi::space, __tree);

		// Fill references
		for (std::string str : params)
			__refs.push_back(nf_zero());

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

			std::cout << "refs:" << std::endl;
			for (auto &ref : __refs)
				ref.print();
		} else {
			std::cout << "Parsing failed" << std::endl;
		}
		
		std::cout << "-------------------------\nstr: " << str <<
			std::endl;
	}

	// Copy constructor and operator
	template <class T, class U>
	node_manager <T, U> ::node_manager(const node_manager &other) :
		__barn(other.__barn), __tree(other.__tree),
		__refs(other.__refs), __params(other.__params)
	{
		rereference(__tree);
	}

	template <class T, class U> node_manager <T, U> &node_manager <T, U>
		::operator=(const node_manager &other)
	{
		if (this != &other) {
			__barn = other.__barn;
			__tree = other.__tree;
			__refs = other.__refs;
			__params = other.__params;

			rereference(__tree);
		}

		return *this;
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

		node *unrefd;
		
		token *tptr;
		node_reference *t;

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
		case token::ndr:
			t = (dynamic_cast <node_reference *>
					(tree.__tptr.get()));

			unrefd = (dynamic_cast <node_reference *>
					(tree.__tptr.get()))->get();

			return unrefd->__tptr.get();
		}

		return nullptr;
	}

	template <class T, class U>
	token *node_manager <T, U> ::substitute_and_compute(std::vector <token *>
			&toks)
	{
		assert(__refs.size() == toks.size());
		for (size_t i = 0; i < __refs.size(); i++) {
			__refs[i] = node(toks[i], {});

			std::cout << "TOKEN TO BE PUT: " << toks[i] << std::endl;
			std::cout << "\tTOKEN TO BE PUT: " << toks[i]->str() << std::endl;
			std::cout << "REF:" << std::endl;
			__refs[i].print();

			label(__refs[i]);
		}

		return value(__tree);
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

			auto itr = find(__params.begin(), __params.end(), tmp);

			size_t index = std::distance(__params.begin(), itr);

			if (itr != __params.end()) {
				out = node(new operation_holder("*"), out,
						node(new node_reference(&__refs[index],
								tmp), {}));

				tmp.clear();
			}

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
	
	template <class T, class U>
	void node_manager <T, U> ::rereference(node &ref)
	{
		if (ref.__tptr->caller() == token::ndr) {
			std::string tmp = (dynamic_cast <node_reference *> (ref.__tptr.get()))->symbol();

			auto itr = find(__params.begin(), __params.end(), tmp);

			size_t index = std::distance(__params.begin(), itr);

			ref.__tptr.reset(new node_reference(&__refs[index], tmp));
		}

		for (node &leaf : ref.__leaves)
			rereference(leaf);
	}

	// Node factories
	template <class T, class U>
	node node_manager <T, U> ::nf_one()
	{
		return node(new operand <U> (1), {});
	}
	
	template <class T, class U>
	node node_manager <T, U> ::nf_zero()
	{
		return node(new operand <U> (0), {});
	}

}

#endif
