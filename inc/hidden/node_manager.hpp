#ifndef NODE_MANAGER_H_
#define NODE_MANAGER_H_

// C/C++ headers
#include <fstream>
#include <stack>

// Engine headers
#include <barn.hpp>
#include <node_reference.hpp>
#include <parser.hpp>
#include <types.hpp>
#include <wildcard.hpp>

namespace zhetapi {

	template <class T, class U>
	class Engine;

	template <class T, class U>
	class Function;

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
		 * Responsible for expanding variable clusters and truning them
		 * into product of operands.
		 */
		void expand(node &);

		void simplify(Engine <T, U> &);

		void differentiate(const std::string &);

		void refactor_reference(const std::string &, token *);

		/*
		 * Code generator. Requires the specification of an output file.
		 */
		void generate(std::string &) const;

		std::string display() const;

		void print(bool = false) const;

		// Static methods
		static bool loose_match(const node_manager <T, U> &, const node_manager <T, U> &);
	private:
		token *value(node) const;
		
		void label(node &);
		void label_operation(node &);
		
		void rereference(node &);

		node expand(const std::string &, const std::vector <node> &);

		void simplify(node &, Engine <T, U> &);

		void differentiate(node &);
		
		void refactor_reference(node &, const std::string &, token *);

		std::string generate(std::string, node, std::ofstream &, size_t &, size_t &) const;

		std::string display(node) const;
		std::string display_operation(node) const;
		std::string display_pemdas(node, node) const;

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

		node *unrefd;
		
		token *tptr;
		
		switch (*(tree.__tptr)) {
		case token::opd:
			return tree.__tptr.get();
		case token::oph:
			for (node leaf : tree.__leaves)
				values.push_back(value(leaf));

			tptr = __barn.compute((dynamic_cast <operation_holder *>
						(tree.__tptr.get()))->rep, values);

			return tptr;
		case token::var:
			tptr = (dynamic_cast <Variable <T, U> *> (tree.__tptr.get()))->get().get();

			return tptr;
		case token::ftn:
			for (node leaf : tree.__leaves)
				values.push_back(value(leaf));

			tptr = (*(dynamic_cast <Function <T, U> *> (tree.__tptr.get())))(values);

			return tptr;
		case token::ndr:
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

			variable_cluster *vclptr = dynamic_cast
				<variable_cluster *> (ref.__tptr.get());

			ref = expand(vclptr->__cluster, ref.__leaves);
		}

		for (node &leaf : ref.__leaves)
			expand(leaf);
	}

	template <class T, class U>
	node node_manager <T, U> ::expand(const std::string &str, const std::vector <node> &leaves)
	{
		typedef std::vector <std::pair <std::vector <node>, std::string>> ctx;
			
		ctx contexts;

		contexts.push_back({{}, ""});

		using namespace std;
		for (size_t i = 0; i < str.length(); i++) {
			ctx tmp;

			for (auto &pr : contexts) {
				pr.second += str[i];
			
				auto itr = find(__params.begin(), __params.end(), pr.second);

				size_t index = std::distance(__params.begin(), itr);

				token *tptr = __barn.get(pr.second);

				bool matches = true;

				node t;
				if (__barn.present(pr.second))
					t = node(new operation_holder(pr.second), {});
				else if (itr != __params.end())
					t = node(new node_reference(&__refs[index], pr.second, index, true), {});
				else if (tptr != nullptr)
					t = node(tptr, {});
				else
					matches = false;

				if (matches) {
					tmp.push_back(pr);

					pr.first.push_back(t);
					pr.second.clear();
				}
			}

			for (auto pr : tmp)
				contexts.push_back(pr);
		}

		/*
                 * Extract the optimal choice. This decision is made based on
                 * the number of tokens read. The heurestic used chooses a
                 * node list which has undergone complete parsing (no leftover
                 * string), and whose size is minimal.
                 */
		std::vector <node> choice;

		bool valid = false;
		for (auto pr : contexts) {
			if (pr.second.empty()) {
				valid = true;

				if (choice.size() == 0)
					choice = pr.first;
				else if (choice.size() > pr.first.size())
					choice = pr.first;
			}
		}
		
		/*
		 * If tmp is not empty, it implies that we could not find a
		 * match for it, and therefore the parsing is incomplete.
		 */
		if (!valid)
			throw undefined_symbol("Undefined symbol cluster \"" + str + "\"");

		/*
		 * The very last token is attributed the leaves
		 */
		choice[choice.size() - 1].__leaves = leaves;

		/* 
		 * Binary fusing. Advantageous to linear fusing in the way in
		 * which it produces a tree with fewer multiplication nodes.
		 */
		while (choice.size() > 1) {
			std::vector <node> tmp;

			size_t n = choice.size();

			for (size_t i = 0; i < n/2; i++) {
				tmp.push_back(node(new operation_holder("*"),
							{choice[i], choice[i +
							1]}));
			}

			if (n % 2)
				tmp.push_back(choice[n - 1]);
		
			choice = tmp;
		}


		return choice[0];
	}

	// Simplication methods
	template <class T, class U>
	void node_manager <T, U> ::simplify(Engine <T, U> &eng)
	{
		using namespace std;

		cout << "Simplifying..." << endl;

		for (auto itr = eng.begin(); itr != eng.end(); itr++) {
			itr->first.print();

			cout << "Should be" << endl;

			itr->second.print();
		}
	}

	template <class T, class U>
	void node_manager <T, U> ::simplify(node &tree, Engine <T, U> &eng)
	{
		using namespace std;

		cout << "Simplifying..." << endl;
	}

	// Differentiation
	template <class T, class U>
	void node_manager <T, U> ::differentiate(const std::string &str)
	{
		for (size_t i = 0; i < __refs.size(); i++) {
			if (__params[i] == str)
				__refs[i].__label = l_variable;
			else
				__refs[i].__label = l_variable_constant;
		}

		label(__tree);

		print();

		std::cout << "\n---------------\n" << std::endl;

		differentiate(__tree);

		print();
	}

	// Post-label usage
	template <class T, class U>
	void node_manager <T, U> ::differentiate(node &ref)
	{
		if (is_constant(ref.__label)) {
			ref.transfer(nf_zero());

			return;
		}

		operation_holder *ophptr = dynamic_cast <operation_holder *> (ref.__tptr.get());

		switch (ophptr->code) {
		case add:
		case sub:
			differentiate(ref.__leaves[0]);
			differentiate(ref.__leaves[1]);
			break;
		}
	}

	// Refactoring methods
	template <class T, class U>
	void node_manager <T, U> ::refactor_reference(const std::string &str, token *tptr)
	{
		refactor_reference(__tree, str, tptr);
	}

	template <class T, class U>
	void node_manager <T, U> ::refactor_reference(node &ref, const
			std::string &str, token *tptr)
	{
		node_reference *ndr = dynamic_cast <node_reference *> (ref.__tptr.get());
		
		if (ndr && ndr->symbol() == str)
			ref.__tptr.reset(tptr);

		for (node &leaf : ref.__leaves)
			refactor_reference(leaf, str, tptr);
	}

	// Generation methods
	template <class T, class U>
	void node_manager <T, U> ::generate(std::string &name) const
	{
		std::ofstream fout(name + ".cpp");

		fout << "#include <token.hpp>\n";
		fout << "#include <function.hpp>\n";
		fout << "\n";

		fout << "extern \"C\" {\n";
		// Make more robust to T and U
		fout << "\tzhetapi::Barn <double, int> " << name << "_barn;\n";

		fout << "\n";
		fout << "\tzhetapi::token *" << name << "(";

		for (size_t i = 0; i < __refs.size(); i++) {
			fout << "zhetapi::token *in" << (i + 1);

			if (i < __refs.size() - 1)
				fout << ", ";
		}

		// print();

		fout << ")\n";
		fout << "\t{\n";

		// Counters
		size_t const_count = 1;
		size_t inter_count = 1;

		// Inside the function
		std::string ret = generate(name, __tree, fout, const_count, inter_count);

		fout << "\t\treturn " << ret << ";\n";
		fout << "\t}\n";
		fout << "}" << std::endl;
	}

	template <class T, class U>
	std::string node_manager <T, U> ::generate(std::string name, node ref,
			std::ofstream &fout, size_t &const_count, size_t
			&inter_count) const
	{
		std::vector <std::string> idents;

		for (auto leaf : ref.__leaves)
			idents.push_back(generate(name, leaf, fout, const_count, inter_count));
		
		if (is_constant_operand(ref.__label)) {
			fout << "\t\tzhetapi::token *c" << const_count++ << " = ";
			fout << "new zhetapi::operand <"
				<< types <T, U> ::proper_symbol(typeid(*(ref.__tptr.get())))
				<< "> (" << ref.__tptr->str() << ");\n";

			return "c" + std::to_string(const_count - 1);
		} else if (ref.__tptr->caller() == token::ndr) {
			node_reference *ndr = dynamic_cast <node_reference *> (ref.__tptr.get());

			return "in" + std::to_string(ndr->index() + 1);
		} else {
			// Assuming we have an operation
			operation_holder *ophtr = dynamic_cast <operation_holder *> (ref.__tptr.get());

			fout << "\t\tzhetapi::token *inter" << inter_count++ <<
				" = " << name << "_barn.compute(\"" <<
				ophtr->rep << "\", {";

			for (size_t i = 0; i < idents.size(); i++) {
				fout << idents[i];

				if (i < idents.size() - 1)
					fout << ", ";
			}

			fout << "});\n";

			return "inter" + std::to_string(inter_count - 1);
		}
	}

	// Displaying utilities
	template <class T, class U>
	std::string node_manager <T, U> ::display() const
	{
		return display(__tree);
	}

	template <class T, class U>
	std::string node_manager <T, U> ::display(node ref) const
	{
		switch (ref.__tptr->caller()) {
		case token::opd:
			return ref.__tptr->str();
		case token::oph:
			return display_operation(ref);
		case token::ndr:
			if ((dynamic_cast <node_reference *> (ref.__tptr.get()))->is_variable())
				return (dynamic_cast <node_reference *> (ref.__tptr.get()))->symbol();
			
			return display(*(dynamic_cast <node_reference *> (ref.__tptr.get())->get()));
		}

		return "?";
	}

	template <class T, class U>
	std::string node_manager <T, U> ::display_operation(node ref) const
	{
		std::string str = (dynamic_cast <operation_holder *> (ref.__tptr.get()))->rep;
		
		operation_holder *ophptr = dynamic_cast <operation_holder *> (ref.__tptr.get());

		switch (ophptr->code) {
		case add:
		case sub:
		case mul:
		case dvs:
			return display_pemdas(ref, ref.__leaves[0]) + " "
				+ str + " " + display_pemdas(ref, ref.__leaves[1]);
		case pwr:
			return display_pemdas(ref, ref.__leaves[0]) + str
				+ display_pemdas(ref, ref.__leaves[1]);
		case sin:
		case cos:
		case tan:

		case xln:
		case xlg:
			return str + "(" + display_pemdas(ref, ref.__leaves[0]) + ")";
		}

		return str;
	}

	template <class T, class U>
	std::string node_manager <T, U> ::display_pemdas(node ref, node child) const
	{
		operation_holder *ophptr = dynamic_cast <operation_holder *> (child.__tptr.get());

		if (!ophptr)
			return display(child);

		switch (ophptr->code) {
		case mul:
		case dvs:
			if ((ophptr->code == add) || (ophptr->code == sub))
				return display(child);
			
			return display(child);
		case pwr:
			if ((ophptr->code == add) || (ophptr->code == sub)
				|| (ophptr->code == mul) || (ophptr->code == dvs))
				return "(" + display(child) + ")";
			
			return display(child);
		}
		
		return display(child);
	}

	// Printing utilities
	template <class T, class U>
	void node_manager <T, U> ::print(bool address) const
	{
		std::cout << "Tree:" << std::endl;

		node_reference::address = address;

		if (address)
			__tree.print();
		else	
			__tree.print_no_address();

		if (__refs.size()) {
			std::cout << "Refs [" << __refs.size() << "]" << std::endl;
			
			for (auto &ref : __refs) {
				if (address)
					ref.print();
				else
					ref.print_no_address();
			}
		}
	}

	// Labeling utilities
	template <class T, class U>
	void node_manager <T, U> ::label(node &ref)
	{
		switch (ref.__tptr->caller()) {
		case token::opd:
			ref.__label = constant_label <T, U> (ref.__tptr.get());
			break;
		case token::oph:
			for (node &leaf : ref.__leaves)
				label(leaf);

			label_operation(ref);

			break;
		case token::ftn:
			/* Also add a different labeling if it is constant,
			 * probably needs to be callde an operation constant
			 */
			ref.__label = l_function;

			for (node &leaf : ref.__leaves)
				label(leaf);

			break;
		case token::ndr:
			// Transfer labels, makes things easier
			ref.__label = (dynamic_cast <node_reference *>
					(ref.__tptr.get()))->get()->__label;
			break;
		}
	}

	template <class T, class U>
	void node_manager <T, U> ::label_operation(node &ref)
	{
		operation_holder *ophptr = dynamic_cast <operation_holder *> (ref.__tptr.get());

		bool constant = true;
		for (auto child : ref.__leaves) {
			if (!is_constant(child.__label)) {
				constant = false;
				break;
			}
		}

		if (constant) {
			ref.__label = l_operation_constant;
			return;
		}

		switch (ophptr->code) {
		case add:
			ref.__label = l_addition;
			return;
		case sub:
			ref.__label = l_subtraction;
			return;
		}
	}
	
	template <class T, class U>
	void node_manager <T, U> ::rereference(node &ref)
	{
		if (ref.__tptr->caller() == token::ndr) {
			std::string tmp = (dynamic_cast <node_reference *> (ref.__tptr.get()))->symbol();

			auto itr = find(__params.begin(), __params.end(), tmp);

			size_t index = std::distance(__params.begin(), itr);

			ref.__tptr.reset(new node_reference(&__refs[index], tmp, index, true));
		}

		for (node &leaf : ref.__leaves)
			rereference(leaf);
	}
	
	// Static methods
	template <class T, class U>
	bool node_manager <T, U> ::loose_match(const node_manager <T, U> &a, const node_manager <T, U> &b)
	{
		return node::loose_match(a.__tree, b.__tree);
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
