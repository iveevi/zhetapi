#ifndef NODE_MANAGER_H_
#define NODE_MANAGER_H_

// C/C++ headers
#include <fstream>
#include <stack>

// Engine headers
#include <barn.hpp>

#include <core/node_reference.hpp>
#include <core/parser.hpp>
#include <core/types.hpp>

namespace zhetapi {

	template <class T, class U>
	class Engine;

	template <class T, class U>
	class Function;

	template <class T, class U>
	class node_manager {
		Barn <T, U>			__barn;
		node				__tree;
		::std::vector <node>		__refs;
		::std::vector <::std::string>	__params;
	public:
		node_manager();
		node_manager(const ::std::string &, Barn <T, U> = Barn <T, U> ());
		node_manager(const ::std::string &, const ::std::vector
				<::std::string> &, Barn <T, U> = Barn <T, U> ());

		node_manager(const node_manager &);

		node_manager &operator=(const node_manager &);

		Token *value(size_t = 0) const;

		Token *substitute_and_compute(::std::vector <Token *> &, size_t = 1);

		/*
		 * Responsible for expanding variable clusters and truning them
		 * into product of Operands.
		 */
		void expand(node &);

		void simplify();

		void differentiate(const ::std::string &);

		void refactor_reference(const ::std::string &, Token *);

		/*
		 * Code generator. Requires the specification of an output file.
		 */
		void generate(::std::string &) const;

		::std::string display() const;

		void print(bool = false) const;

		// Static methods
		static bool loose_match(const node_manager <T, U> &, const node_manager <T, U> &);
	private:
		Token *value(node, size_t = 0) const;

		size_t count_up(node &);
		
		void label(node &);
		void label_operation(node &);
		
		void rereference(node &);

		node expand(const ::std::string &, const ::std::vector <node> &);

		void simplify(node &);
		void simplify_separable(node &);

		void differentiate(node &);
		void differentiate_mul(node &);
		void differentiate_pow(node &);
		void differentiate_ln(node &);
		void differentiate_lg(node &);
		void differentiate_const_log(node &);
		void differentiate_trig(node &);
		void differentiate_hyp(node &);
		
		void refactor_reference(node &, const ::std::string &, Token *);

		::std::string generate(::std::string, node, ::std::ofstream &, size_t &, size_t &) const;

		::std::string display(node) const;
		::std::string display_operation(node) const;
		::std::string display_pemdas(node, node) const;

		/*
		 * Node factories; produce special nodes such as ones, zeros,
		 * etc. to make constuction of such nodes easy.
		 */
		static node nf_one();
		static node nf_zero();
	public:
		// General error
		class error {
			::std::string str;
		public:
			error(::std::string s) : str(s) {}

			const ::std::string &what() const {
				return str;
			}
		};

		// Syntax error
		class syntax_error : public error {
		public:
			syntax_error(::std::string s) : error(s) {}
		};

		// Undefined symbol error
		class undefined_symbol : public error {
		public:
			undefined_symbol(::std::string s) : error(s) {}
		};
	};

	template <class T, class U>
	node_manager <T, U> ::node_manager() {}

	template <class T, class U>
	node_manager <T, U> ::node_manager(const ::std::string &str, Barn <T, U>
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
		count_up(__tree);
	}

	template <class T, class U>
	node_manager <T, U> ::node_manager(const ::std::string &str, const
			::std::vector <::std::string> &params, Barn <T, U> barn) :
		__params(params), __barn(barn) 
	{
		zhetapi::parser <T, U> pr;

		siter iter = str.begin();
		siter end = str.end();

		bool r = qi::phrase_parse(iter, end, pr, qi::space, __tree);

		// Fill references
		node tmp;
		for (::std::string str : params) {
			tmp = nf_zero();

			tmp.__label = l_variable;

			__refs.push_back(tmp);
		}

		// Unpack variable clusters
		expand(__tree);

		// Label the tree
		label(__tree);
		count_up(__tree);
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
	Token *node_manager <T, U> ::value(size_t spare_threads) const
	{
		return value(__tree, spare_threads);
	}

	template <class T, class U>
	Token *node_manager <T, U> ::value(node tree, size_t spare_threads) const
	{
		::std::vector <Token *> values;
		node *unrefd;
		Token *tptr;
		int size;

		::std::cout << "Spare threads: " << spare_threads << ::std::endl;
		
		switch (*(tree.__tptr)) {
		case Token::opd:
			return tree.__tptr.get()->copy();
		case Token::oph:
			size = tree.__leaves.size();
			for (node leaf : tree.__leaves)
				values.push_back(value(leaf));

			tptr = __barn.compute((dynamic_cast <operation_holder *>
						(tree.__tptr.get()))->rep, values);

			return tptr->copy();
		case Token::var:
			tptr = (dynamic_cast <Variable <T, U> *> (tree.__tptr.get()))->get().get();

			return tptr->copy();
		case Token::ftn:
			if (tree.__leaves.empty())
				return tree.__tptr->copy();
			
			for (node leaf : tree.__leaves)
				values.push_back(value(leaf));

			tptr = (*(dynamic_cast <Function <T, U> *> (tree.__tptr.get())))(values);

			return tptr->copy();
		case Token::ndr:
			unrefd = (dynamic_cast <node_reference *>
					(tree.__tptr.get()))->get();

			return unrefd->__tptr.get()->copy();
		case Token::reg:
			for (node leaf : tree.__leaves)
				values.push_back(value(leaf));

			tptr = (*(dynamic_cast <Registrable *> (tree.__tptr.get())))(values);

			return tptr->copy();
		}

		return nullptr;
	}

	template <class T, class U>
	Token *node_manager <T, U> ::substitute_and_compute(::std::vector <Token *>
			&toks, size_t total_threads)
	{
		assert(__refs.size() == toks.size());
		for (size_t i = 0; i < __refs.size(); i++) {
			__refs[i] = node(toks[i], {});

			label(__refs[i]);
		}

		return value(__tree, total_threads - 1);
	}

	// Expansion methods
	template <class T, class U>
	void node_manager <T, U> ::expand(node &ref)
	{
		if (*(ref.__tptr) == Token::vcl) {
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
	node node_manager <T, U> ::expand(const ::std::string &str, const ::std::vector <node> &leaves)
	{
		typedef ::std::vector <::std::pair <::std::vector <node>, ::std::string>> ctx;
			
		ctx contexts;

		contexts.push_back({{}, ""});

		using namespace std;
		for (size_t i = 0; i < str.length(); i++) {
			ctx tmp;

			for (auto &pr : contexts) {
				pr.second += str[i];
			
				auto itr = find(__params.begin(), __params.end(), pr.second);

				size_t index = ::std::distance(__params.begin(), itr);

				Token *tptr = __barn.get(pr.second);

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
                 * the number of Tokens read. The heurestic used chooses a
                 * node list which has undergone complete parsing (no leftover
                 * string), and whose size is minimal.
                 */
		::std::vector <node> choice;

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
		 * The very last Token is attributed the leaves
		 */
		choice[choice.size() - 1].__leaves = leaves;

		/*
		 * Binary fusing. Advantageous to linear fusing in the way in
		 * which it produces a tree with fewer multiplication nodes.
		 */
		while (choice.size() > 1) {
			::std::vector <node> tmp;

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

	// Counting nodes
	template <class T, class U>
	size_t node_manager <T, U> ::count_up(node &ref)
	{
		size_t total = 1;
		for (auto &child : ref.__leaves)
			total += count_up(child);
		
		ref.__nodes = total;

		return total;
	}

	// Simplication methods
	template <class T, class U>
	void node_manager <T, U> ::simplify()
	{
		simplify(__tree);
	}

	template <class T, class U>
	void node_manager <T, U> ::simplify(node &ref)
	{
		if (ref.__label == l_operation_constant) {
			ref.transfer(node(value(ref), l_constant, {}));

			return;
		}

		operation_holder *ophptr = dynamic_cast <operation_holder *> (ref.__tptr.get());

		// if (ophptr && (ophptr->code == add || ophptr->code == sub))
		//	simplify_separable(ref);

		for (auto &child : ref.__leaves)
			simplify(child);
	}

	template <class T, class U>
	void node_manager <T, U> ::simplify_separable(node &ref)
	{
		Token *opd = new Operand <U> (0);

		::std::stack <node> process;

		::std::vector <node> sums;
		
		process.push(ref);

		node top;
		while (!process.empty()) {
			top = process.top();

			process.pop();

			operation_holder *ophptr = dynamic_cast <operation_holder *> (top.__tptr.get());

			if (ophptr && (ophptr->code == add || ophptr->code == sub)) {
				process.push(top.__leaves[0]);
				process.push(top.__leaves[1]);
			} else {
				sums.push_back(top);
			}
		}

		// Make copies, not address copies
		::std::vector <node> rest;
		for (auto sep : sums) {
			// Includes variable_constant; remove this exception
			if (is_constant(sep.__label))
				opd = __barn.compute("+", {opd, value(sep)});
			else
				rest.push_back(sep);
		}

		if (!types <T, U> ::is_zero(opd))
			rest.push_back(node(opd, {}));

		while (rest.size() > 1) {
			::std::vector <node> tmp;

			size_t n = rest.size();

			for (size_t i = 0; i < n - 1; i += 2) {
				tmp.push_back(node(new operation_holder("+"),
							{rest[i], rest[i + 1]}));
			}

			if (n % 2)
				tmp.push_back(rest[n - 1]);
		
			rest = tmp;
		}

		ref.transfer(rest[0]);
	}

	// Differentiation
	template <class T, class U>
	void node_manager <T, U> ::differentiate(const ::std::string &str)
	{
		for (size_t i = 0; i < __refs.size(); i++) {
			if (__params[i] == str)
				__refs[i].__label = l_variable;
			else
				__refs[i].__label = l_variable_constant;
		}

		label(__tree);

		differentiate(__tree);

		simplify();
	}

	// Post-label usage
	template <class T, class U>
	void node_manager <T, U> ::differentiate(node &ref)
	{
		if (is_constant(ref.__label)) {
			ref.transfer(nf_zero());

			return;
		}

		switch (ref.__label) {
		case l_separable:
			differentiate(ref.__leaves[0]);
			differentiate(ref.__leaves[1]);
			break;
		case l_multiplied:
			differentiate_mul(ref);
			break;
		case l_power:
			differentiate_pow(ref);
			break;
		case l_natural_log:
			differentiate_ln(ref);
			break;
		case l_binary_log:
			differentiate_lg(ref);
			break;
		case l_constant_base_log:
			differentiate_const_log(ref);
			break;
		case l_trigonometric:
			differentiate_trig(ref);
			break;
		case l_hyperbolic:
			differentiate_hyp(ref);
			break;
		case l_variable:
			ref.transfer(nf_one());
			break;
		}
	}

	// Refactoring methods
	template <class T, class U>
	void node_manager <T, U> ::refactor_reference(const ::std::string &str, Token *tptr)
	{
		refactor_reference(__tree, str, tptr);
	}

	template <class T, class U>
	void node_manager <T, U> ::refactor_reference(node &ref, const
			::std::string &str, Token *tptr)
	{
		node_reference *ndr = dynamic_cast <node_reference *> (ref.__tptr.get());
		
		if (ndr && ndr->symbol() == str)
			ref.__tptr.reset(tptr);

		for (node &leaf : ref.__leaves)
			refactor_reference(leaf, str, tptr);
	}

	// Generation methods
	template <class T, class U>
	void node_manager <T, U> ::generate(::std::string &name) const
	{
		::std::ofstream fout(name + ".cpp");

		fout << "#include <token.hpp>\n";
		fout << "#include <function.hpp>\n";
		fout << "\n";

		fout << "extern \"C\" {\n";
		// Make more robust to T and U
		fout << "\tzhetapi::Barn <double, int> " << name << "_barn;\n";

		fout << "\n";
		fout << "\tzhetapi::Token *" << name << "(";

		for (size_t i = 0; i < __refs.size(); i++) {
			fout << "zhetapi::Token *in" << (i + 1);

			if (i < __refs.size() - 1)
				fout << ", ";
		}

		fout << ")\n";
		fout << "\t{\n";

		// Counters
		size_t const_count = 1;
		size_t inter_count = 1;

		// Inside the function
		::std::string ret = generate(name, __tree, fout, const_count, inter_count);

		fout << "\t\treturn " << ret << ";\n";
		fout << "\t}\n";
		fout << "}" << ::std::endl;
	}

	template <class T, class U>
	::std::string node_manager <T, U> ::generate(::std::string name, node ref,
			::std::ofstream &fout, size_t &const_count, size_t
			&inter_count) const
	{
		::std::vector <::std::string> idents;

		for (auto leaf : ref.__leaves)
			idents.push_back(generate(name, leaf, fout, const_count, inter_count));
		
		if (is_constant_operand(ref.__label)) {
			fout << "\t\tzhetapi::Token *c" << const_count++ << " = ";
			fout << "new zhetapi::Operand <"
				<< types <T, U> ::proper_symbol(typeid(*(ref.__tptr.get())))
				<< "> (" << ref.__tptr->str() << ");\n";

			return "c" + ::std::to_string(const_count - 1);
		} else if (ref.__tptr->caller() == Token::ndr) {
			node_reference *ndr = dynamic_cast <node_reference *> (ref.__tptr.get());

			return "in" + ::std::to_string(ndr->index() + 1);
		} else {
			// Assuming we have an operation
			operation_holder *ophtr = dynamic_cast <operation_holder *> (ref.__tptr.get());

			fout << "\t\tzhetapi::Token *inter" << inter_count++ <<
				" = " << name << "_barn.compute(\"" <<
				ophtr->rep << "\", {";

			for (size_t i = 0; i < idents.size(); i++) {
				fout << idents[i];

				if (i < idents.size() - 1)
					fout << ", ";
			}

			fout << "});\n";

			return "inter" + ::std::to_string(inter_count - 1);
		}
	}

	// Displaying utilities
	template <class T, class U>
	::std::string node_manager <T, U> ::display() const
	{
		return display(__tree);
	}

	template <class T, class U>
	::std::string node_manager <T, U> ::display(node ref) const
	{
		switch (ref.__tptr->caller()) {
		case Token::opd:
			return ref.__tptr->str();
		case Token::oph:
			return display_operation(ref);
		case Token::ndr:
			if ((dynamic_cast <node_reference *> (ref.__tptr.get()))->is_variable())
				return (dynamic_cast <node_reference *> (ref.__tptr.get()))->symbol();
			
			return display(*(dynamic_cast <node_reference *> (ref.__tptr.get())->get()));
		}

		return "?";
	}

	template <class T, class U>
	::std::string node_manager <T, U> ::display_operation(node ref) const
	{
		::std::string str = (dynamic_cast <operation_holder *> (ref.__tptr.get()))->rep;
		
		operation_holder *ophptr = dynamic_cast <operation_holder *> (ref.__tptr.get());

		switch (ophptr->code) {
		case add:
		case sub:
		case mul:
			return display_pemdas(ref, ref.__leaves[0]) + " "
				+ str + " " + display_pemdas(ref, ref.__leaves[1]);
		case dvs:
			return display_pemdas(ref, ref.__leaves[0])
				+ str + display_pemdas(ref, ref.__leaves[1]);
		case pwr:
			return display_pemdas(ref, ref.__leaves[0]) + str
				+ display_pemdas(ref, ref.__leaves[1]);
		case sin:
		case cos:
		case tan:
		case sec:
		case csc:
		case cot:

		case snh:
		case csh:
		case tnh:
		case cch:
		case sch:
		case cth:

		case xln:
		case xlg:
			return str + "(" + display_pemdas(ref, ref.__leaves[0]) + ")";
		case lxg:
			return str + "(" + display_pemdas(ref, ref.__leaves[0])
				+ ", " + display_pemdas(ref, ref.__leaves[1]) + ")";
		}

		return str;
	}

	template <class T, class U>
	::std::string node_manager <T, U> ::display_pemdas(node ref, node child) const
	{
		operation_holder *rophptr = dynamic_cast <operation_holder *> (ref.__tptr.get());
		operation_holder *cophptr = dynamic_cast <operation_holder *> (child.__tptr.get());

		if (!cophptr)
			return display(child);

		switch (rophptr->code) {
		case mul:
			if ((cophptr->code == add) || (cophptr->code == sub))
				return display(child);
			
			return display(child);
		case dvs:
			return "(" + display(child) + ")";
		case pwr:
			if ((cophptr->code == add) || (cophptr->code == sub)
				|| (cophptr->code == mul) || (cophptr->code == dvs))
				return "(" + display(child) + ")";
			
			return display(child);
		}
		
		return display(child);
	}

	// Printing utilities
	template <class T, class U>
	void node_manager <T, U> ::print(bool address) const
	{
		node_reference::address = address;

		if (address)
			__tree.print();
		else	
			__tree.print_no_address();

		if (__refs.size()) {
			::std::cout << "Refs [" << __refs.size() << "]" << ::std::endl;
			
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
		case Token::opd:
			ref.__label = constant_label <T, U> (ref.__tptr.get());
			break;
		case Token::oph:
			for (node &leaf : ref.__leaves)
				label(leaf);

			label_operation(ref);

			break;
		case Token::ftn:
			/* Also add a different labeling if it is constant,
			 * probably needs to be called an operation constant
			 */
			ref.__label = l_function;

			for (node &leaf : ref.__leaves)
				label(leaf);

			break;
		case Token::ndr:
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
		case sub:
			ref.__label = l_separable;
			break;
		case mul:
			ref.__label = l_multiplied;
			break;
		case dvs:
			ref.__label = l_divided;
			break;
		case pwr:
			if ((ref.__leaves[0].__label == l_variable)
				&& (is_constant(ref.__leaves[1].__label)))
				ref.__label = l_power;
			break;
		case xln:
			ref.__label = l_natural_log;
			break;
		case xlg:
			ref.__label = l_binary_log;
			break;
		case lxg:
			if (is_constant(ref.__leaves[0].__label) &&
				!is_constant(ref.__leaves[1].__label))
				ref.__label = l_constant_base_log;
			break;
		case sin:
		case cos:
		case tan:
		case sec:
		case csc:
		case cot:
			ref.__label = l_trigonometric;
			break;
		case snh:
		case csh:
		case tnh:
		case cch:
		case sch:
		case cth:
			ref.__label = l_hyperbolic;
			break;
		}
	}
	
	template <class T, class U>
	void node_manager <T, U> ::rereference(node &ref)
	{
		if (ref.__tptr->caller() == Token::ndr) {
			::std::string tmp = (dynamic_cast <node_reference *> (ref.__tptr.get()))->symbol();

			auto itr = find(__params.begin(), __params.end(), tmp);

			size_t index = ::std::distance(__params.begin(), itr);

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
		return node(new Operand <U> (1), {});
	}
	
	template <class T, class U>
	node node_manager <T, U> ::nf_zero()
	{
		return node(new Operand <U> (0), {});
	}

}

#include <core/node_differentiation.hpp>

#endif