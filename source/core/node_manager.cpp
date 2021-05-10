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

// Operators
node &node_manager::operator[](size_t i)
{
	return _tree[i];
}

const node &node_manager::operator[](size_t i) const
{
	return _tree[i];
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

// Value finding methods
Token *node_manager::value(Engine *context) const
{
	return node_value(context, _tree);
}

// Sequential value (returns null for now)
Token *node_manager::sequential_value(Engine *context) const
{
	return node_sequential_value(context, _tree);
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

	return node_value(context, _tree);
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

// Setters
void node_manager::set_label(lbl label)
{
	_tree._label = label;
}

// Branch compress/translation
void node_manager::compress_branches()
{
	compress_branches(_tree);
}

void node_manager::create_branch(node &tree, size_t start, size_t end)
{
	node_manager branch;

	auto itr = tree.begin() + start;
	for (size_t i = start; i < end; i++) {
		branch.append(tree[start]);

		tree._leaves.erase(itr);
	}

	branch.set_label(l_branch);

	tree._leaves.insert(itr, branch._tree);
}

// Make this non-method
void node_manager::compress_branches(node &tree)
{
	if (tree.label() != l_sequential)
		return;
	
	// Add a dummy node to flush
	tree.append(node());
	
	int n;
	int start;

	start = -1;
	
	n = tree.child_count();
	for (size_t i = 0; i < n; i++) {
		switch (tree[i].label()) {
		case l_if_branch:
			if (start != -1)
				create_branch(tree, start, i);

			start = i;
			break;
		case l_else_branch:
		case l_elif_branch:
			// Make sure that stsrt != -1
			break;
		// TODO: Add the rest of the branch types here
		default:
			if (start != -1)
				create_branch(tree, start, i);
			
			start = -1;
			break;
		}
	}

	// Remove the flushed
	tree.remove_end();
}

// Appending nodes
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

void node_manager::remove_end()
{
	_tree.remove_end();
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

		if (ophptr && ophptr->code == attribute) {
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
	if (ophptr && ophptr->code == attribute)
		attribute_invert(ref);

	if (ref.caller() == Token::vcl) {
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
	if (ref.label() == l_operation_constant) {
		ref.transfer(node(node_value(context, ref), l_constant, {}));

		return;
	}

	operation_holder *ophptr = ref.cast <operation_holder> ();

	if (ophptr && (ophptr->code == add || ophptr->code == sub)) {
		// Fix subtraction and what not
		simplify_separable(context, ref, ophptr->code);
	} else if (ophptr && (ophptr->code == mul || ophptr->code == dvs)) {
		simplify_mult_div(context, ref, ophptr->code);
	} else if (ophptr && (ophptr->code == pwr)) {
		simplify_power(context, ref);
	} else {
		for (auto &child : ref._leaves)
			simplify(context, child);
	}
}

void node_manager::simplify_separable(Engine *context, node &ref, codes c)
{
	Token *opd = new OpZ(0);
	Token *zero = new OpZ(0);		// Make static or make a function is_zero
	Token *true_token = new OpB(true);

	std::stack <std::pair <bool, node>> process;

	std::vector <node> plus;
	std::vector <node> minus;

	process.push({true, ref});

	std::pair <bool, node> top;
	node topn;

	while (!process.empty()) {
		top = process.top();
		topn = top.second;

		process.pop();

		operation_holder *ophptr = topn.cast <operation_holder> ();

		// Add is_separable code check
		if (ophptr && (ophptr->code == add || ophptr->code == sub)) {			
			process.push({top.first, topn[0]});
			process.push({
				(ophptr->code == add) ? top.first : !top.first,
				topn[1]
			});
		} else {
			if (top.first)
				plus.push_back(topn);
			else
				minus.push_back(topn);
		}
	}

	size_t i;
	
	i = 0;
	while (i < plus.size()) {
		if (is_constant(plus[i].label())) {
			opd = context->compute("+", {
				opd,
				node_value(context, plus[i])
			});

			plus.erase(plus.begin() + i);
		} else {
			i++;
		}
	}

	i = 0;
	while (i < minus.size()) {
		if (is_constant(minus[i].label())) {
			opd = context->compute("-", {
				opd,
				node_value(context, minus[i])
			});

			minus.erase(minus.begin() + i);
		} else {
			i++;
		}
	}

	if (!tokcmp(opd, zero)) {
		Token *tptr = context->compute(">", {opd, zero});
		Token *nopd = context->compute("*", {opd, new OpZ(-1)});

		if (tokcmp(tptr, true_token))
			plus.push_back(node(opd));
		else
			minus.push_back(node(nopd));
	}

	// Next step is to fold the vectors
	while (plus.size() > 1) {
		std::vector <node> tmp;

		size_t n = plus.size();

		for (size_t i = 0; i < n/2; i++) {
			tmp.push_back(
				node(new operation_holder("+"),
					{
						plus[i],
						plus[i + 1]
					}
				)
			);
		}

		if (n % 2)
			tmp.push_back(plus[n - 1]);
	
		plus = tmp;
	}

	while (minus.size() > 1) {
		std::vector <node> tmp;

		size_t n = minus.size();

		for (size_t i = 0; i < n/2; i++) {
			tmp.push_back(
				node(new operation_holder("+"),
					{
						minus[i],
						minus[i + 1]
					}
				)
			);
		}

		if (n % 2)
			tmp.push_back(minus[n - 1]);
	
		minus = tmp;
	}

	node all;
	
	if (!plus.empty()) {
		if (!minus.empty()) {
			all.retokenize(new operation_holder("-"));

			all.append(plus[0]);
			all.append(minus[0]);
		} else {
			all = plus[0];
		}
	} else {
		if (!minus.empty()) {
			all.retokenize(new operation_holder("*"));

			all.append(node(new OpZ(-1)));
			all.append(minus[0]);
		}
	}
	
	ref.transfer(all);

	// Simplify rest
	operation_holder *ophptr = ref.cast <operation_holder> ();

	if (ophptr->code == add || ophptr->code == sub) {
		simplify(context, ref[0]);
		simplify(context, ref[1]);
	} else {
		simplify(context, ref);
	}
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

	// Mult/div simplification
	Token *opd = new OpZ(1);
	Token *onez = new OpZ(1);		// TODO: Make static or make a function is_one
	Token *oneq = new OpQ(1);
	Token *oner = new OpR(1);
	Token *true_token = new OpB(true);

	std::stack <std::pair <bool, node>> process;

	std::vector <node> mult;
	std::vector <node> divs;

	process.push({true, ref});

	std::pair <bool, node> top;
	node topn;

	while (!process.empty()) {
		top = process.top();
		topn = top.second;

		process.pop();

		operation_holder *ophptr = topn.cast <operation_holder> ();

		// Add is_separable code check
		if (ophptr && (ophptr->code == mul || ophptr->code == dvs)) {			
			process.push({top.first, topn[0]});
			process.push({
				(ophptr->code == mul) ? top.first : !top.first,
				topn[1]
			});
		} else {
			if (top.first)
				mult.push_back(topn);
			else
				divs.push_back(topn);
		}
	}

	size_t i;
	
	i = 0;
	while (i < mult.size()) {
		if (is_constant(mult[i].label())) {
			opd = context->compute("*", {
				opd,
				node_value(context, mult[i])
			});

			mult.erase(mult.begin() + i);
		} else {
			i++;
		}
	}

	i = 0;
	while (i < divs.size()) {
		if (is_constant(divs[i].label())) {
			opd = context->compute("/", {
				opd,
				node_value(context, divs[i])
			});

			divs.erase(divs.begin() + i);
		} else {
			i++;
		}
	}

	if (!tokcmp(opd, onez)
		&& !tokcmp(opd, oneq)
		&& !tokcmp(opd, oner))
		mult.push_back(node(opd));

	// Next step is to fold the vectors
	// TODO: make as a helper function, with central operation as 2nd input
	while (mult.size() > 1) {
		std::vector <node> tmp;

		size_t n = mult.size();

		for (size_t i = 0; i < n/2; i++) {
			tmp.push_back(
				node(new operation_holder("*"),
					{
						mult[i],
						mult[i + 1]
					}
				)
			);
		}

		if (n % 2)
			tmp.push_back(mult[n - 1]);
	
		mult = tmp;
	}

	while (divs.size() > 1) {
		std::vector <node> tmp;

		size_t n = divs.size();

		for (size_t i = 0; i < n/2; i++) {
			tmp.push_back(
				node(new operation_holder("*"),
					{
						divs[i],
						divs[i + 1]
					}
				)
			);
		}

		if (n % 2)
			tmp.push_back(divs[n - 1]);
	
		divs = tmp;
	}

	node all;
	
	if (!mult.empty()) {
		if (!divs.empty()) {
			all.retokenize(new operation_holder("/"));

			all.append(mult[0]);
			all.append(divs[0]);
		} else {
			all = mult[0];
		}
	} else {
		if (!divs.empty()) {
			all.retokenize(new operation_holder("^"));

			all.append(node(new OpZ(-1)));
			all.append(divs[0]);
		}
	}
	
	ref.transfer(all);

	// Simplify rest
	operation_holder *ophptr = ref.cast <operation_holder> ();

	if (ophptr && (ophptr->code == mul || ophptr->code == dvs)) {
		simplify(context, ref[0]);
		simplify(context, ref[1]);
	} else {
		simplify(context, ref);
	}
}

void node_manager::simplify_power(Engine *context, node &ref)
{
	Token *one = new OpZ(1);
	Token *zero = new OpZ(0);

	Token *tptr = ref[1].ptr();

	if (tokcmp(tptr, one)) {
		ref = ref[0];

		simplify(context, ref);
	} else if (tokcmp(tptr, zero)) {
		ref = node(one->copy());

		simplify(context, ref);
	} else {
		simplify(context, ref[0]);
		simplify(context, ref[1]);
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
	label(_tree);
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
		return ref._tptr->dbg_str();
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
	case sub:
		if (cophptr->code == add)
			return "(" + display(child) + ")";
		
		return display(child); 
	case mul:
		// TODO: What is this?
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
	return node(new OpZ(1), {});
}

node node_manager::nf_zero()
{
	return node(new OpZ(0), {});
}

}
