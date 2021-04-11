#include <engine.hpp>

#include <core/common.hpp>
#include <core/node_manager.hpp>

namespace zhetapi {

// Static variables
Engine *node_manager::shared_context = new Engine();

node_manager::node_manager() {}

node_manager::node_manager(const node_manager &other)
		: _tree(other._tree), _refs(other._refs),
		_params(other._params)
{
	rereference(_tree);
}

node_manager::node_manager(Engine *context, const node &tree)
		: _tree(tree)
{
	// Unpack variable clusters
	expand(context, _tree);

	// Label the tree
	label(_tree);
	count_up(_tree);

	// Simplify
	simplify(context, _tree);
}

node_manager::node_manager(Engine *context, const node &tree, const Args &args)
		: _tree(tree), _params(args)
{
	// TODO: Put this in a method
	//
	// Fill references
	node tmp;
	for (std::string str : args) {
		tmp = nf_zero();

		tmp._label = l_variable;

		_refs.push_back(tmp);
	}

	// Unpack variable clusters
	expand(context, _tree);

	// Label the tree
	label(_tree);
	count_up(_tree);
	
	rereference(_tree);

	// Simplify
	simplify(context, _tree);
}

node_manager::node_manager(Engine *context, const std::string &str)
{
	zhetapi::parser pr;

	siter iter = str.begin();
	siter end = str.end();

	bool r = qi::phrase_parse(iter, end, pr, qi::space, _tree);

	/* using namespace std;
	cout << "TREE PRE PROCESSED:" << endl;
	_tree.print(); */

	// Unpack variable clusters
	expand(context, _tree);

	// Label the tree
	label(_tree);
	count_up(_tree);
	
	// Simplify
	simplify(context, _tree);
}

node_manager::node_manager(
		Engine *context,
		const std::string &str,
		const std::vector <std::string> &params,
		const std::set <std::string> &pardon)
		: _params(params)
{
	parser pr;

	siter iter = str.begin();
	siter end = str.end();

	bool r = qi::phrase_parse(iter, end, pr, qi::space, _tree);

	// Fill references
	node tmp;
	for (std::string str : params) {
		tmp = nf_zero();

		tmp._label = l_variable;

		_refs.push_back(tmp);
	}

	// Unpack variable clusters
	expand(context, _tree, pardon);

	// Label the tree
	label(_tree);
	count_up(_tree);

	// Simplify
	simplify(context, _tree);
}

node_manager &node_manager::operator=(const node_manager &other)
{
	if (this != &other) {
		shared_context = other.shared_context;
		_tree = other._tree;
		_refs = other._refs;
		_params = other._params;

		rereference(_tree);
	}

	return *this;
}

// Properties
bool node_manager::empty() const
{
	return _tree.empty();
}

size_t node_manager::num_args() const
{
	return _params.size();
}

const node &node_manager::get_tree() const
{
	return _tree;
}

// Setters
void node_manager::set_label(lbl label)
{
	_tree._label = label;
}

// Value finding methods
Token *node_manager::value(Engine *context) const
{
	return value(context, _tree);
}

// Sequential value (returns null for now)
Token *node_manager::sequential_value(Engine *context) const
{
	// Assumes that the top node is a sequential
	for (node nd : _tree._leaves)
		value(context, nd);
	
	return nullptr;
}

Token *node_manager::value(Engine *context, node tree) const
{
	std::vector <Token *> values;

	Token *tptr;
	Token *vptr;

	Variable v;

	rvalue *rv;

	algorithm *aptr;

	std::string ident;

	int size;

	// If null token, resort to special execution modes
	// (Use this instead of sequential_value for
	// algorithms; dont remove it though)
	//
	// Special nodes!
	if (tree.null()) {
		if (tree.label() == l_assignment_chain) {
			// Evaluate first node
			Token *tmp = value(context, tree[0]);

			// Assign for the other nodes

			// Add index operator for nodes
			size_t nleaves = tree.child_count(); // Use a method instead

			for (size_t i = 1; i < nleaves; i++) {
				// Ensure that the node has type lvalue
				if (tree[i].label() != l_lvalue)
					throw std::runtime_error("Need an lvalue on the left side of an \'=\'");
				
				lvalue *lv = tree[i].cast <lvalue> ();

				lv->assign(tmp, context);
			}

			return nullptr;
		} else {
			throw std::runtime_error("Unknown execution mode \'" + strlabs[tree._label] + "\'");
		}
	}

	operation_holder *ophptr = tree.cast <operation_holder> ();

	if (ophptr && ophptr->code == atm) {
		lvalue *lv = tree[1].cast <lvalue> ();

		// TODO: throw on nullptr
		std::string at = lv->symbol();

		std::vector <Token *> args;

		for (node leaf : tree[1]._leaves)
			args.push_back(value(context, leaf));

		Token *callee = tree[0]._tptr;

		if (tree[0].child_count() > 0)
			callee = value(context, tree[0]);

		return callee->attr(at, args);
	}

	// else: this func
	
	// TODO: Add a method for nodes to cast the token (ie. tree.cast <type> ())
	
	switch (tree.caller()) {
	case Token::opd:
		return tree.copy_token();
	case Token::oph:	
		size = tree._leaves.size();

		// TODO: add begin and end for trees
		for (node leaf : tree._leaves)
			values.push_back(value(context, leaf));

		tptr = context->compute((tree.cast <operation_holder> ())->rep, values);
		
		if (tree._label == l_post_modifier) {
			rv = tree[0].cast <rvalue> ();
			
			context->put(rv->symbol(), tptr);

			return rv->get(context)->copy();
		} else if (tree._label == l_pre_modifier) {
			rv= tree[0].cast <rvalue> ();

			context->put(rv->symbol(), tptr);
		}

		return tptr->copy();
	case Token::var:
		return (tree.cast <Variable> ())->get()->copy();
	case Token::token_rvalue:
		return (tree.cast <rvalue> ())->get(context)->copy();
	case Token::ndr:
		return (tree.cast <node_reference> ())->get()->copy_token();
	case Token::token_node_list:
		return (tree.cast <node_list> ())->evaluate(context);
	case Token::ftn:
		if (tree._leaves.empty())
			return tree._tptr->copy();
		
		for (node leaf : tree._leaves)
			values.push_back(value(context, leaf));

		tptr = (*(dynamic_cast <Function *> (tree._tptr)))(values);

		return tptr->copy();
	case Token::reg:
		for (node leaf : tree._leaves)
			values.push_back(value(context, leaf));

		tptr = (*(dynamic_cast <Registrable *> (tree._tptr)))(values);

		if (tptr)
			return tptr->copy();

		break;
	case Token::alg:
		for (node leaf : tree._leaves)
			values.push_back(value(context, leaf));
		
		aptr = dynamic_cast <algorithm *> (tree._tptr);
		tptr = aptr->execute(context, values);

		if (tptr)
			return tptr->copy();

		break;
	default:
		break;
	}

	return nullptr;
}

Token *node_manager::substitute_and_compute(
		Engine *context,
		std::vector <Token *> &toks)
{
	assert(_refs.size() == toks.size());
	for (size_t i = 0; i < _refs.size(); i++) {
		_refs[i] = node(toks[i], {});

		label(_refs[i]);
	}

	return value(context, _tree);
}

Token *node_manager::substitute_and_seq_compute(
		Engine *context,
		const std::vector <Token *> &toks)
{
	assert(_refs.size() == toks.size());
	for (size_t i = 0; i < _refs.size(); i++) {
		_refs[i] = node(toks[i], {});

		label(_refs[i]);
	}

	return sequential_value(context);
}

void node_manager::append(const node &n)
{
	_tree.append(n);

	// Add the rest of the elements
	count_up(_tree);
}

void node_manager::append(const node_manager &nm)
{
	_tree.append(nm._tree);

	// Add the rest of the elements
	count_up(_tree);
}

// Better names
void node_manager::append_front(const node &n)
{
	_tree.append_front(n);

	// Add the rest of the elements
	count_up(_tree);
}

void node_manager::append_front(const node_manager &nm)
{
	_tree.append_front(nm._tree);

	// Add the rest of the elements
	count_up(_tree);
}

void node_manager::add_args(const std::vector <std::string> &args)
{
	// Fill references
	node tmp;
	for (std::string str : args) {
		tmp = nf_zero();

		tmp._label = l_variable;

		_refs.push_back(tmp);
	}
	
	// Add the arguments
	_params.insert(_params.end(), args.begin(), args.end());

	// Fix broken variable references
	rereference(_tree);
}

// Unpacking methods
void node_manager::unpack()
{
	unpack(_tree);

	// TODO: only relable those that were unpacked
	// needs a separate method to label singletons
	// without dfs-ing
	label(_tree);
}

void node_manager::unpack(node &ref)
{
	if (ref.caller() == Token::token_rvalue) {
		// TODO: make a method for assigning tokens to trees
		ref._tptr = (ref.cast <rvalue> ())->get(shared_context);
	}

	// Add a foreach method in nodes (with ref)
	for (node &nd : ref._leaves)
		unpack(nd);
}

// Expansion methods
void attribute_invert(node &ref)
{
	std::vector <node> unfolded {ref[0]};

	node current = ref[1];
	while (true) {
		operation_holder *ophptr = current.cast <operation_holder> ();

		if (ophptr && ophptr->code == atm) {
			// Reduce variable clusters immediately
			if (current[0].caller() == Token::vcl) {
				variable_cluster *vclptr = current[0].cast <variable_cluster> ();
				lvalue *lv = new lvalue(vclptr->_cluster);
				current[0].retokenize(lv);

				delete lv;
			}

			unfolded.push_back(current[0]);

			current = current[1];
		} else {
			// Reduce variable clusters immediately
			if (current.caller() == Token::vcl) {
				variable_cluster *vclptr = current.cast <variable_cluster> ();
				lvalue *lv = new lvalue(vclptr->_cluster);
				current.retokenize(lv);

				delete lv;
			}

			unfolded.push_back(current);

			break;
		}
	}

	/* using namespace std;
	cout << string(50, '=') << endl;
	cout << "UNFODLED:" << endl;
	for (auto nd : unfolded)
		nd.print();
	cout << string(50, '=') << endl; */

	operation_holder *atmptr = new operation_holder(".");

	while (unfolded.size() > 1) {
		unfolded[0] = node(atmptr->copy(), {
			unfolded[0],
			unfolded[1]
		});

		unfolded.erase(std::next(unfolded.begin()));
	}

	ref = unfolded[0];

	// Free resources
	delete atmptr;
}

void node_manager::expand(
		Engine *context,
		node &ref,
		const std::set <std::string> &pardon)
{
	operation_holder *ophptr = ref.cast <operation_holder> ();
	if (ophptr && ophptr->code == atm)
		attribute_invert(ref);

	if (ref._tptr->caller() == Token::vcl) {
		/*
		 * Excluding the parameters, the variable cluster should
		 * always be a leaf of the tree.
		 */

		variable_cluster *vclptr = ref.cast <variable_cluster> ();

		ref = expand(context, vclptr->_cluster, ref._leaves, pardon);
	}

	for (node &leaf : ref._leaves)
		expand(context, leaf, pardon);
}

node node_manager::expand(
		Engine *context,
		const std::string &str,
		const std::vector <node> &leaves,
		const std::set <std::string> &pardon)
{
	typedef std::vector <std::pair <std::vector <node>, std::string>> ctx;
		
	ctx contexts;

	contexts.push_back({{}, ""});

	// Check once for differential
	Token *dtok = context->get("d");
	auto ditr = std::find(_params.begin(), _params.end(), "d");

	for (size_t i = 0; i < str.length(); i++) {
		ctx tmp;

		for (auto &pr : contexts) {
			pr.second += str[i];
		
			auto itr = std::find(_params.begin(), _params.end(), pr.second);

			size_t index = std::distance(_params.begin(), itr);

			Token *tptr = context->get(pr.second);

			// TODO: get the alias of pr.second (clean this code ;-;)
			// Potential differential node
			Token *dptr = nullptr;
			auto diff = _params.end();
			if ((pr.second)[0] == 'd'
				&& (dtok == nullptr)
				&& (ditr == _params.end())) {
				// Priority on parameters
				diff = find(_params.begin(), _params.end(), pr.second.substr(1));
				dptr = context->get(pr.second.substr(1));

				// Second chance for differential
				if (!dptr && !context->get(pr.second))
					dptr = new lvalue(pr.second.substr(1));
			}

			size_t dindex = std::distance(_params.begin(), diff);

			bool matches = true;

			node t;
			if (context->present(pr.second)) {
				t = node(new operation_holder(pr.second), {});
			} else if (itr != _params.end()) {
				t = node(new node_reference(&_refs[index], pr.second, index, true), {});
			} else if (tptr != nullptr) {
				// Delaying actual evaluation to
				// evaluation - better for algorithms,
				// where values are not always known
				// for sure
				//
				// TODO: Add special case for base scope,
				// where dependencies can be ignored (does
				// not include if/else statements)
				//
				// TODO: Add a block class (maybe oversees
				// the algorithm class as well)
				
				// For now only operands are allowed as rvalues
				if (tptr->caller() == Token::opd) {
					rvalue *rv = new rvalue(pr.second);

					t = node(rv);
				} else {
					// t = node(new rvalue(pr.second, shared_context), {});
					t = node(tptr);
				}
			} else if (diff != _params.end()) {
				t = node(new node_differential(new node_reference(&_refs[dindex], pr.second.substr(1), dindex, true)));
			} else if (dptr != nullptr) {
				t = node(new node_differential(dptr));
			} else if (pardon.find(pr.second) != pardon.end()) {
				// Use the pardon set as a last resort
				t = node(new rvalue(pr.second));
			} else {
				matches = false;
			}

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
		throw undefined_symbol(str);

	/*
	 * The very last Token is attributed the leaves
	 */
	choice[choice.size() - 1]._leaves = leaves;

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

// Counting nodes
size_t node_manager::count_up(node &ref)
{
	size_t total = 1;
	for (auto &child : ref._leaves)
		total += count_up(child);
	
	ref._nodes = total;

	return total;
}

// Simplication methods
void node_manager::simplify(Engine *context)
{
	simplify(context, _tree);
}

void node_manager::simplify(Engine *context, node &ref)
{
	if (ref._label == l_operation_constant) {
		ref.transfer(node(value(context, ref), l_constant, {}));

		return;
	}

	operation_holder *ophptr = ref.cast <operation_holder> ();

	if (ophptr && (ophptr->code == add || ophptr->code == sub)) {
		// Fix subtraction and what not
		// simplify_separable(ref);
	} else if (ophptr && (ophptr->code == mul || ophptr->code == dvs)) {
		simplify_mult_div(context, ref, ophptr->code);
	} else {
		for (auto &child : ref._leaves)
			simplify(context, child);
	}
}

void node_manager::simplify_separable(Engine *context, node &ref)
{
	Token *opd = new opd_z(0);
	Token *zero = new opd_z(0);

	std::stack <node> process;

	std::vector <node> sums;
	
	process.push(ref);

	node top;
	while (!process.empty()) {
		top = process.top();

		process.pop();

		operation_holder *ophptr = top.cast <operation_holder> ();

		if (ophptr && (ophptr->code == add || ophptr->code == sub)) {
			process.push(top[0]);
			process.push(top[1]);
		} else {
			sums.push_back(top);
		}
	}

	std::vector <node> variables;
	
	std::stack <node> constants;
	for (auto nm : sums) {
		if (is_constant(nm.label()))
			constants.push(nm);
		else
			variables.push_back(nm);
	}
	
	while (!constants.empty()) {
		node nd = constants.top();

		constants.pop();

		opd = context->compute("+", {opd, nd._tptr});
	}

	// Still needs to deal with variables
	std::vector <node> all = variables;

	if (!tokcmp(opd, zero))
		all.push_back(node(opd));

	// Next step is to fold the vector
	while (all.size() > 1) {
		std::vector <node> tmp;

		size_t n = all.size();

		for (size_t i = 0; i < n/2; i++) {
			tmp.push_back(
				node(new operation_holder("+"),
					{
						all[i],
						all[i + 1]
					}
				)
			);
		}

		if (n % 2)
			tmp.push_back(all[n - 1]);
	
		all = tmp;
	}

	ref.transfer(all[0]);
}

void node_manager::simplify_mult_div(Engine *context, node &ref, codes c)
{
	if (c == dvs) {
		lbl l1 = ref._leaves[0]._label;
		lbl l2 = ref._leaves[1]._label;

		if (l1 == l_differential && l2 == l_differential) {
			Token *t1 = ref._leaves[0]._tptr;
			Token *t2 = ref._leaves[1]._tptr;

			t1 = (dynamic_cast <node_differential *> (t1))->get();
			t2 = (dynamic_cast <node_differential *> (t2))->get();

			Function *ftn = nullptr;
			std::string var;

			if (t1->caller() == Token::ftn)
				ftn = dynamic_cast <Function *> (t1);
			
			// TODO: add a token base for tokens with a .symbol() method
			if (t2->caller() == Token::ndr)
				var = (dynamic_cast <node_reference *> (t2))->symbol();
			else if (t2->caller() == Token::token_lvalue)
				var = (dynamic_cast <lvalue *> (t2))->symbol();
			
			if (ftn && ftn->is_variable(var)) {
				Function f = ftn->differentiate(var);

				ref._leaves.clear();
				ref._tptr = f.copy();
			}
		}
	}
}

// Differentiation
void node_manager::differentiate(const std::string &str)
{
	for (size_t i = 0; i < _refs.size(); i++) {
		if (_params[i] == str)
			_refs[i]._label = l_variable;
		else
			_refs[i]._label = l_variable_constant;
	}

	label(_tree);

	differentiate(_tree);

	simplify(shared_context);
}

// Post-label usage
void node_manager::differentiate(node &ref)
{
	if (is_constant(ref._label)) {
		ref.transfer(nf_zero());

		return;
	}

	switch (ref._label) {
	case l_separable:
		differentiate(ref._leaves[0]);
		differentiate(ref._leaves[1]);
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
	default:
		break;
	}
}

// Refactoring methods
void node_manager::refactor_reference(const std::string &str, Token *tptr)
{
	refactor_reference(_tree, str, tptr);
}

void node_manager::refactor_reference(
		node &ref,
		const std::string &str,
		Token *tptr)
{
	node_reference *ndr = dynamic_cast <node_reference *> (ref._tptr);
	
	if (ndr && ndr->symbol() == str)
		ref._tptr = tptr->copy();

	for (node &leaf : ref._leaves)
		refactor_reference(leaf, str, tptr);
}

// Displaying utilities
std::string node_manager::display() const
{
	return display(_tree);
}

std::string node_manager::display(node ref) const
{
	switch (ref._tptr->caller()) {
	case Token::opd:
		return ref._tptr->str();
	case Token::oph:
		return display_operation(ref);
	case Token::ndr:
		if ((dynamic_cast <node_reference *> (ref._tptr))->is_variable())
			return (dynamic_cast <node_reference *> (ref._tptr))->symbol();
		
		return display(*(dynamic_cast <node_reference *> (ref._tptr)->get()));
	default:
		break;
	}

	return "?";
}

std::string node_manager::display_operation(node ref) const
{
	std::string str = (dynamic_cast <operation_holder *> (ref._tptr))->rep;
	
	operation_holder *ophptr = dynamic_cast <operation_holder *> (ref._tptr);

	switch (ophptr->code) {
	case add:
	case sub:
	case mul:
		return display_pemdas(ref, ref._leaves[0]) + " "
			+ str + " " + display_pemdas(ref, ref._leaves[1]);
	case dvs:
		return display_pemdas(ref, ref._leaves[0])
			+ str + display_pemdas(ref, ref._leaves[1]);
	case pwr:
		return display_pemdas(ref, ref._leaves[0]) + str
			+ display_pemdas(ref, ref._leaves[1]);
	case sxn:
	case cxs:
	case txn:
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
		return str + "(" + display_pemdas(ref, ref._leaves[0]) + ")";
	case lxg:
		// Fix bug with single/double argument overload
		return str + "(" + display_pemdas(ref, ref._leaves[0])
			+ ", " + display_pemdas(ref, ref._leaves[1]) + ")";
	default:
		break;
	}

	return str;
}

std::string node_manager::display_pemdas(node ref, node child) const
{
	operation_holder *rophptr = dynamic_cast <operation_holder *> (ref._tptr);
	operation_holder *cophptr = dynamic_cast <operation_holder *> (child._tptr);

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
	default:
		break;
	}
	
	return display(child);
}

// Printing utilities
void node_manager::print(bool address) const
{
	node_reference::address = address;

	if (address)
		_tree.print();
	else	
		_tree.print_no_address();

	if (_refs.size()) {
		std::cout << "Refs [" << _refs.size() << "]" << std::endl;
		
		for (auto &ref : _refs) {
			if (address)
				ref.print();
			else
				ref.print_no_address();
		}
	}
}

// Labeling utilities
void node_manager::label(node &ref)
{
	switch (ref.caller()) {
	case Token::opd:
		ref.relabel(constant_label(ref.ptr()));
		break;
	case Token::token_wildcard:
		ref.relabel(l_wildcard);
		break;
	case Token::oph:
		// TODO: Method
		for (node &leaf : ref._leaves)
			label(leaf);

		label_operation(ref);

		break;
	case Token::ftn:
		for (node &leaf : ref._leaves)
			label(leaf);
		
		/* Also add a different labeling if it is constant,
		 * probably needs to be called an operation constant
		 */
		ref._label = l_function;
		break;
	case Token::var:
		ref._label = l_variable;
		break;
	case Token::ndr:
		// Transfer labels, makes things easier
		ref._label = (dynamic_cast <node_reference *>
				(ref._tptr))->get()->_label;
		break;
	case Token::ndd:
		ref._label = l_differential;
		break;
	case Token::reg:
		for (node &leaf : ref._leaves)
			label(leaf);

		ref._label = l_registrable;
		break;
	default:
		break;
	}
}

void node_manager::label_operation(node &ref)
{
	operation_holder *ophptr = ref.cast <operation_holder> ();

	bool constant = true;
	for (auto child : ref._leaves) {
		if (!is_constant(child.label())) {
			constant = false;
			break;
		}
	}

	if (constant) {
		ref.relabel(l_operation_constant);
		return;
	}

	// Add relabelling method
	switch (ophptr->code) {
	case add:
	case sub:
		ref.relabel(l_separable);
		break;
	case mul:
		ref.relabel(l_multiplied);
		break;
	case dvs:
		ref.relabel(l_divided);
		break;
	case pwr:
		if ((ref[0].label() == l_variable)
			&& (is_constant(ref[1].label())))
			ref.relabel(l_power);
		else
			ref.relabel(l_power_misc);
		break;
	case xln:
		ref.relabel(l_natural_log);
		break;
	case xlg:
		ref.relabel(l_binary_log);
		break;
	case lxg:
		if (is_constant(ref._leaves[0]._label) &&
			!is_constant(ref._leaves[1]._label))
			ref.relabel(l_constant_base_log);
		break;
	case sxn:
	case cxs:
	case txn:
	case sec:
	case csc:
	case cot:
		ref.relabel(l_trigonometric);
		break;
	case snh:
	case csh:
	case tnh:
	case cch:
	case sch:
	case cth:
		ref.relabel(l_hyperbolic);
		break;
	case pin:
	case pde:
		ref.relabel(l_post_modifier);
		break;
	case rin:
	case rde:
		ref.relabel(l_pre_modifier);
		break;
	default:
		break;
	}
}

void node_manager::rereference(node &ref)
{
	if (ref._tptr && (ref._tptr->caller() == Token::ndr)) {
		std::string tmp = (dynamic_cast <node_reference *> (ref._tptr))->symbol();

		auto itr = find(_params.begin(), _params.end(), tmp);

		if (itr == _params.end())
			throw std::runtime_error("could not find param " + tmp);

		// TODO: throw if index is at the end
		size_t index = std::distance(_params.begin(), itr);

		// Need a new method to clear/reset
		ref._tptr = new node_reference(&_refs[index], tmp, index, true);
	}

	for (node &leaf : ref._leaves)
		rereference(leaf);
}

// Arithmetic
node_manager operator+(const node_manager &a, const node_manager &b)
{
	// TODO: Add a union operation for Engines
	return node_manager(
		node_manager::shared_context,
		node(new operation_holder("+"), {
			a.get_tree(),
			b.get_tree()
		}),
		args_union(a._params, b._params)
	);
}

node_manager operator-(const node_manager &a, const node_manager &b)
{
	// TODO: Add a method to make this a one-liner
	return node_manager(
		node_manager::shared_context,
		node(new operation_holder("-"), {
			a.get_tree(),
			b.get_tree()
		}),
		args_union(a._params, b._params)
	);
}

// Static methods
bool node_manager::loose_match(const node_manager &a, const node_manager &b)
{
	return node::loose_match(a._tree, b._tree);
}

// Node factories
node node_manager::nf_one()
{
	return node(new opd_z(1), {});
}

node node_manager::nf_zero()
{
	return node(new opd_z(0), {});
}

}
