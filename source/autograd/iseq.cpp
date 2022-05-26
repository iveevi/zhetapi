#include "../../include/autograd/iseq.hpp"

namespace zhetapi {

namespace autograd {

//////////////////////
// Public interface //
//////////////////////

// Empty constructor
ISeq::ISeq() : _function(0, op_iseq) {}

// Get variable
_variable *ISeq::get(int index) const
{
	return _vars[index];
}

// Inserting instructions and functions
void ISeq::append(const _function *fptr)
{
	// Append, then optimize
	_append(fptr);
	_optimize();
}

// Evaluate the sequence
Constant ISeq::compute(const Input &ins) const
{
	// Load inputs
	_load(ins);

	// Stack of operands
	std::stack <Constant> ops;

	// Execute all the instructions
	for (const _function *ftn : _instrs)
		_exec(ftn, ops);

	// Return top-most value on the stack
	return ops.top();
}

// Make copy
_function *ISeq::copy() const
{
	return new ISeq(_instrs, _consts, inputs);
}

// Dump instructions for debugging
std::string ISeq::summary() const
{
	// Headers
	io::Args args {
		"Index",
		"Instruction",
		"Variable",
		"Constant"
	};

	// Fill out the rows
	std::vector <io::Args> rows;
	for (int i = 0; i < _instrs.size(); i++) {
		io::Args row;

		row.push_back(std::to_string(i));
		row.push_back(_instrs[i]->summary());

		if (i < _vars.size())
			row.push_back(_vars[i]->summary());
		else
			row.push_back("");

		std::ostringstream oss;

		// TODO: should print shape, not value
		if (i < _consts.size()) {
			oss << _consts[i];
			row.push_back(oss.str());
			// row.push_back("--shape--");
		} else
			row.push_back("");

		rows.push_back(row);
	}

	// Return formatted table
	return io::table(args, rows);
}

/////////////////////////
// Protected interface //
/////////////////////////

// Kernel function and # of inputs
ISeq::ISeq(const _function *ftn, int nins)
		: _function(nins, op_iseq)
{
	// Fill the _vars with variables
	_vars.resize(nins);

	for (int i = 0; i < ftn->inputs; i++)
		_vars[i] = new _variable();

	// Add the get instructions
	_instrs.resize(nins + 1);

	int i = 0;
	for (; i < nins; i++)
		_instrs[i] = new Get(i);

	// Add the kernel
	_instrs[i] = ftn;
}

// List of instructions and constants
ISeq::ISeq(std::vector <const _function *> instrs,
	std::vector <Constant> consts, int nins)
		: _function(nins, op_iseq),
		_instrs(instrs),
		_consts(consts)
{
	// Fill the _vars with variables
	_vars.resize(nins);

	for (int i = 0; i < nins; i++)
		_vars[i] = new _variable();
}

ISeq::ISeq(std::vector <const _function *> instrs,
	std::vector <Constant> consts, int nins,
	const _reindex_map &reindex)
		: _function(nins, op_iseq),
		_instrs(instrs),
		_consts(consts)
{
	// Fill the _vars with variables
	_vars.resize(nins);

	for (int i = 0; i < nins; i++)
		_vars[i] = new _variable(reindex.at(i));
}

//////////////////////
// Append functions //
//////////////////////

// Append helpers
void ISeq::_append(const _function *fptr)
{
	switch (fptr->spop) {
	case op_var:
		append_variable((_variable *) fptr);
		break;
	case op_iseq:
		append_iseq((ISeq *) fptr);
		break;
	default:
		// Just add the function to the instructions
		_instrs.push_back(fptr);
		break;
	}
}

void ISeq::append_variable(_variable *v)
{
	// Check if the variable exists already
	int index = index_of(v);

	// Add the variable (and increment inputs) if not found
	int gindex = index;
	if (index < 0) {
		gindex = _vars.size();
		_vars.push_back(v);
		inputs++;
	}

	_instrs.push_back(new Get(gindex));
}

void ISeq::append_iseq(ISeq *iseq)
{
	for (const _function *fptr : iseq->_instrs) {
		_function *nptr = fptr->copy();
		if (nptr->spop == op_get) {
			// TODO: clean up
			int i = reinterpret_cast <Get *> (nptr)->index;
			_variable *v = iseq->_vars[i];

			// TODO: should create a new value as well
			i = index_of(v);

			if (i == -1) {
				// Add a variable
				i = _vars.size();
				_vars.push_back((_variable *) v->copy());
				inputs++;
			}

			_append(new Get (i));
			continue;
		} else if (nptr->spop == op_const) {
			int index = reinterpret_cast <const Const *> (nptr)->index;
			Constant c = iseq->_consts[index];

			// Add the constant to this ISeq
			_consts.push_back(c);
			_append(new Const(_consts.size() - 1));
		}

		_append(nptr);
	}
}

// Get index of variable
int ISeq::index_of(_variable *v)
{
	int index = -1;
	for (int i = 0; i < _vars.size(); i++) {
		if (v->id == _vars[i]->id) {
			index = i;
			break;
		}
	}

	return index;
}

/////////////////////////
// Computation helpers //
/////////////////////////

// Get constant from the stack and handle any errors
static Constant _get_operand(std::stack <Constant> &ops)
{
	// Check empty first
	if (ops.empty())
		throw std::runtime_error("ISeq::_get_operand: empty stack");

	Constant c = ops.top();
	ops.pop();

	return c;
}

// Operation kernels
using Kernel = Constant (*)(std::stack <Constant> &);

// List of kernels
static std::vector <Kernel> kernels = {
	// Addition
	[](std::stack <Constant> &ops) -> Constant {
		Constant b = _get_operand(ops);
		Constant a = _get_operand(ops);
		return a + b;
	},

	// Subtraction
	[](std::stack <Constant> &ops) -> Constant {
		Constant b = _get_operand(ops);
		Constant a = _get_operand(ops);
		return a - b;
	},

	// Multiplication
	[](std::stack <Constant> &ops) -> Constant {
		Constant b = _get_operand(ops);
		Constant a = _get_operand(ops);

		// TODO: use matrix multiplication if dims == 2
		// or if dim== 2 and dim == 1 -> user must do hammard
		// for element-wise multiplication
		return multiply(a, b);
	},

	// Division
	[](std::stack <Constant> &ops) -> Constant {
		Constant b = _get_operand(ops);
		Constant a = _get_operand(ops);
		return divide(a, b);
	}
};

// Load inputs
void ISeq::_load(const Input &ins) const
{
	// Ensure correct number of inputs
	if (ins.size() != inputs) {
		std::stringstream ss;
		ss << "ISeq::_load: expected " << inputs << " inputs, got " << ins.size();
		throw std::runtime_error(ss.str());
	}

	for (int i = 0; i < ins.size(); i++)
		_vars[i]->value = ins[i];
}

// Storing a constant into the cache
void ISeq::storec(std::stack <Constant> &ops, int i) const
{
	// Get from the stack
	Constant c = _get_operand(ops);

	if (i >= _cache.size())
		_cache.resize(i + 1);

	_cache[i] = c;
}

// Deal with special instructions
bool ISeq::_ispec(const _function *ftn, std::stack <Constant> &ops) const
{
	int index;
	switch (ftn->spop) {
	case op_get:
		// Get index and push the corresponding variable
		index = reinterpret_cast <const Get *> (ftn)->index;
		ops.push(_vars[index]->value.copy());
		return true;
	case op_const:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const Const *> (ftn)->index;
		ops.push(_consts[index].copy());
		return true;
	case op_store_cache:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const _store_cache *> (ftn)->index;
		storec(ops, index);
		return true;
	case op_get_cache:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const _get_cache *> (ftn)->index;
		ops.push(_cache[index].copy());
		return true;
	case op_add: case op_sub:
	case op_mul: case op_div:
		// Push the result of the operation
		ops.push(kernels[ftn->spop - op_add](ops).copy());
		return true;
	default:
		break;
	}

	return false;
}

// Execute an instruction
void ISeq::_exec(const _function *ftn, std::stack <Constant> &ops) const
{
	// Check if the instruction is a special operation
	if (_ispec(ftn, ops)) {
		// std::cout << "Special op (" << ftn->summary() << "), value = " << ops.top() << std::endl;
		return;
	}

	// Ensure that the stack has enough operands
	if (ops.size() < ftn->inputs) {
		throw std::runtime_error("ISeq::_exec: not enough operands");
	}	// TODO: throw, except if #inputs = -1 (variadic)

	// Load the inputs
	Input fins;
	for (int i = 0; i < ftn->inputs; i++) {
		Constant c = ops.top();
		ops.pop();

		fins.push_back(c);
	}

	// Evaluate the instruction
	Constant n = ftn->compute(fins);

	// std::cout << "ISEQ: value = " << n << std::endl;
	// std::cout << summary() << std::endl;

	// Push new value onto stack
	ops.push(n);
}

//////////////////////////
// Function composition //
//////////////////////////

// Composition helpers
void _compose_iseq(ISeq::Instructions &instrs, const ISeq *iseq,
		const _function *ftn, std::unordered_map <int, int> &reindex,
		int &vindex)
{
	// If the instruction is a get,
	// add the corresponding variable
	if (ftn->spop == _function::op_get) {
		int index = reinterpret_cast <const Get *> (ftn)->index;

		// Get variable id from iseq
		int id = iseq->get(index)->id;

		// Check if the variable is already in reindex map
		if (reindex.find(id) == reindex.end()) {
			// Add the variable to the reindex map
			reindex[id] = vindex;

			// Add the variable to the instruction sequence
			instrs.push_back(new Get(vindex));

			// Increment the variable index
			vindex++;
		} else {
			// Add the variable to the instruction sequence
			instrs.push_back(new Get(reindex[id]));
		}
	} else {
		instrs.push_back(ftn);
	}
}

// Override composition
_function *ISeq::_compose(const Compositions &cs) const
{
	// ISeq variables
	Instructions instrs;
	std::vector <Constant> consts;

	// Free cache offset
	int fcache = cs.size();

	// Variable index
	int vindex = 0;
	
	// Map for reindexing get operations
	//	variable id -> get index
	std::unordered_map <int, int> reindex;

	// Iterate through inputs
	for (int i = 0; i < cs.size(); i++) {
		const _function *ftn = cs[i];

		// Switch variables
		const ISeq *iseq;
		const _repl_const *rc;
		int id;

		// TODO: should not be composed get instructions, only var and
		// iseq

		// TODO: use if statement, not switch
		switch (ftn->spop) {
		case op_var:
			// Keep the variable
			id = reinterpret_cast <const _variable *> (ftn)->id;
			reindex[id] = vindex;
			instrs.push_back(new Get(vindex++));
			break;
		case op_iseq:
			// Get the iseq, and append its instructions
			iseq = reinterpret_cast <const ISeq *> (ftn);

			for (const _function *fptr : iseq->_instrs)
				_compose_iseq(instrs, iseq, fptr, reindex, vindex);
			break;
		case op_repl_const:
			rc = reinterpret_cast <const _repl_const *> (ftn);
			consts.push_back(rc->value);
			instrs.push_back(new Const(rc->index));
			break;
		default:
			// Add the instruction
			instrs.push_back(ftn->copy());
			break;
		}

		// Add store cache instruction
		instrs.push_back(new _store_cache(i));
	}

	// Add own instructions
	for (const _function *ftn : _instrs) {
		int index;

		switch (ftn->spop) {
		case op_get:
			// Changes to get cache
			index = reinterpret_cast <const Get *> (ftn)->index;
			instrs.push_back(new _get_cache(index));
			break;
		default:
			// Add the instruction
			instrs.push_back(ftn->copy());
			break;
		}
	}

	// Invert the reindex map
	_reindex_map inv;

	for (const auto &pair : reindex)
		inv[pair.second] = pair.first;

	// Composition returns a new ISeq
	//	the variables are reused
	//	according to the reindex map
	ISeq *iseq = new ISeq(instrs, consts, vindex, inv);

	// Optimize the ISeq
	iseq->_optimize();
	return iseq;
}

//////////////////////////////////
// Tree building and rebuilding //
//////////////////////////////////

// Constructors for the tree
_node::_node(const _function *f) : fptr(f) {}
_node::_node(const _function *f, const std::vector <_node *> &cs)
		: fptr(f), children(cs) {}

// Get string
std::string _node::str(int indent) const
{
	// Get string of current node
	std::string s = fptr->summary();

	// Indentation
	std::string sindent(indent, '\t');

	// If there are children
	for (const _node *child : children) {
		s += "\n" + sindent + "\u2514\u2500\u2500 "
			+ child->str(indent + 1);
	}

	return s;
}

// Constructors for cache info
ISeq::_cache_info::_cache_info() {}
ISeq::_cache_info::_cache_info(int r, _node *tree)
		: refs(r), value(tree) {}

// Generating the tree
_node *ISeq::_tree(_cache_map &cache) const
{
	// Keep a stack of nodes,
	// 	similar to computation
	std::stack <_node *> nodes;

	// Iterate through the instructions
	for (const _function *ftn : _instrs)
		_tree_walk(ftn, nodes, cache);

	// Return the top node
	return nodes.top();
}

// TODO: static method?
void ISeq::_tree_walk(const _function *ftn, std::stack <_node *> &nodes, _cache_map &cache) const
{
	// Switch variables
	std::vector <_node *> children;
	_node *value;
	int index;

	// Check type of operation
	switch (ftn->spop) {
	case op_get:
		break;
	case op_get_cache:
		// Get index
		index = reinterpret_cast <const _get_cache *> (ftn)->index;

		// Get value from cache map
		value = cache[index].value;

		// Increment the reference count
		cache[index].refs++;

		children = {value};
		break;
	case op_store_cache:
		// Get index of cache
		index = reinterpret_cast <const _get_cache *> (ftn)->index;

		// Insert value in cache map
		value = nodes.top();
		nodes.pop();
		
		cache[index] = _cache_info {0, value};
		children = {value};
		break;
	default:
		// Get number of inputs
		int nins = ftn->inputs;

		// Pop and push the children
		for (int i = 0; i < nins; i++) {
			// Pop the child
			_node *child = nodes.top();
			nodes.pop();

			// Push the child (to the front)
			children.insert(children.begin(), child);
		}

		// Add the node
		break;
	}
	
	// Add the node
	nodes.push(new _node(ftn, children));
}

// Rebuild the ISeq from the tree
void ISeq::_rebuild(const _node *tree, Instructions &instrs,
		ConstantCache &ccache, _cache_map &cache,
		const ConstantCache &pconsts) const
{
	// Iterate through the tree
	const _function *ftn = tree->fptr;

	if (ftn->spop == op_get_cache) {
		// Get the index
		int index = reinterpret_cast <const _get_cache *> (ftn)->index;

		// Get the cache info
		_cache_info info = cache[index];

		// Check possible optimization
		if (info.refs == 1) {
			// Add value instead of get cache
			_rebuild(info.value, instrs, ccache, cache, pconsts);
		} else {
			// Add get cache
			instrs.push_back(ftn);
		}
	} else if (ftn->spop == op_repl_const) {
		// std::cout << "Replacing constant " << reinterpret_cast <const _repl_const *> (ftn)->value << std::endl;

		// Get the constant
		Constant c = reinterpret_cast
			<const _repl_const *> (ftn)->value;

		// TODO: should optimize constants if there are a lot

		// Add the constant
		int index = ccache.size();
		ccache.push_back(c);

		// Add instruction
		instrs.push_back(new Const(index));
	} else if (ftn->spop == op_const) {
		// std::cout << "Constant index = " << reinterpret_cast <const Const *> (ftn)->index << std::endl;
		// std::cout << "\tpconsts size = " << pconsts.size() << std::endl;
		Constant c = pconsts[reinterpret_cast <const Const *> (ftn)->index];

		// Add the constant
		int index = ccache.size();
		ccache.push_back(c);

		// Add instruction
		instrs.push_back(new Const(index));
	} else {
		// Add the operands
		for (const _node *child : tree->children)
			_rebuild(child, instrs, ccache, cache, pconsts);

		// Add the instruction
		instrs.push_back(ftn);
	}
}

/////////////////////////////////////////
// Optimizing the instruction sequence //
/////////////////////////////////////////

// Optimize the ISeq
void ISeq::_optimize()
{
	/* std::cout << "===========================" << std::endl;
	std::cout << "Optimizing ISeq:" << std::endl;
	std::cout << summary() << std::endl;

	std::cout << "\nconsts: " << _consts.size() << std::endl;
	for (int i = 0; i < _consts.size(); i++)
		std::cout << "\t" << i << ": " << _consts[i] << std::endl; */

	_cache_map cache;
	_node *tree = _tree(cache);

	Instructions instrs;
	ConstantCache ccache;
	_rebuild(tree, instrs, ccache, cache, _consts);

	// Replace the instructions and constant cache
	// TODO: clean up excess memory
	_instrs = instrs;
	_consts = ccache;

	// TODO: prune duplicate constants
	/* std::cout << "Optimized ISeq:" << std::endl;
	std::cout << summary() << std::endl;

	std::cout << "\nconsts: " << _consts.size() << std::endl;
	for (int i = 0; i < _consts.size(); i++)
		std::cout << "\t" << i << ": " << _consts[i] << std::endl;
	std::cout << "===========================" << std::endl; */
}


/////////////////////
// Differentiation //
/////////////////////

// Forward declarations
_node *_diff_tree(const _node *, int);

// Differentiation kernels
_node *_diffk_get(const _function *fptr, const _node *tree, int vindex)
{
	// Get the index
	int index = reinterpret_cast <const Get *> (fptr)->index;

	// Check if the index is the same as the variable
	if (index == vindex)
		return new _node(new _repl_const(1, vindex));
	
	// Add the node
	return new _node(new _repl_const(0, vindex));
}

_node *_diffk_add_sub(const _function *fptr, const _node *tree, int vindex)
{
	_function *f = new _function(1, fptr->spop);
	_node *d1 = _diff_tree(tree->children[0], vindex);
	_node *d2 = _diff_tree(tree->children[1], vindex);
	return new _node(f, {d1, d2});
}

_node *_diffk_mul(const _function *fptr, const _node *tree, int vindex)
{
	_function *f1 = new _function(1, _function::op_add);
	_function *f2 = new _function(1, _function::op_mul);

	_node *c1 = tree->children[0];
	_node *c2 = tree->children[1];

	_node *d1 = _diff_tree(c1, vindex);
	_node *d2 = _diff_tree(c2, vindex);

	_node *c3 = new _node(f2, {c1, d2});
	_node *c4 = new _node(f2, {c2, d1});

	return new _node(f1, {c4, c3});
}

_node *_diffk_div(const _function *fptr, const _node *tree, int vindex)
{
	_function *f1 = new _function(1, _function::op_sub);
	_function *f2 = new _function(1, _function::op_mul);
	_function *f3 = new _function(1, _function::op_div);

	_node *c1 = tree->children[0];
	_node *c2 = tree->children[1];

	_node *d1 = _diff_tree(c1, vindex);
	_node *d2 = _diff_tree(c2, vindex);

	_node *c3 = new _node(f2, {c2, d1});
	_node *c4 = new _node(f2, {c1, d2});

	_node *c5 = new _node(f1, {c3, c4});
	_node *c6 = new _node(f2, {c2, c2});

	return new _node(f3, {c5, c6});
}

// Construct tree for special instructions
_node *_diff_ispec(const _function *ftn,
		const _node *tree, int vindex)
{
	// TODO: operation factories (which return shared_ptr)
	
	// TODO: map for kernels?

	// Check type of operation
	switch (ftn->spop) {
	case _function::op_get:
		return _diffk_get(ftn, tree, vindex);
	case _function::op_const:
		return new _node(new _repl_const(0, vindex));
	case _function::op_add:
	case _function::op_sub:
		return _diffk_add_sub(ftn, tree, vindex);
	case _function::op_mul:
		return _diffk_mul(ftn, tree, vindex);
	case _function::op_div:
		return _diffk_div(ftn, tree, vindex);
	default:
		break;
	}

	// Default return
	return nullptr;
}

// Substitute a differentiated template
_node *_diff_iseq(const _node *orig,
		const std::vector <_node *> &ins,
		const std::vector <Constant> &consts,
		int vindex)
{
	// TODO: check that there are no more ISeqs
	const _function *ftn = orig->fptr;

	// Temporary indices
	int index = -1;

	// Any new children
	std::vector <_node *> children;

	// Switch special instructions
	switch (ftn->spop) {
	case _function::op_get:
		// Get the index
		index = reinterpret_cast <const Get *> (ftn)->index;
		return ins[index];
	case _function::op_const:
		// Replace with _repl_const
		index = reinterpret_cast <const Const *> (ftn)->index;
		return new _node(new _repl_const(consts[index], -1));
	case _function::op_differential:
		// Get the index
		index = reinterpret_cast <const _iop *> (ftn)->index;
		return _diff_tree(ins[index], vindex);
	default:
		// Reiterate children
		for (const _node *child : orig->children)
			children.push_back(_diff_iseq(child, ins, consts, vindex));

		// Create new node
		return new _node(ftn, children);
	}

	return nullptr;
}

// Construct a new, differentiated tree
_node *_diff_tree(const _node *tree, int vindex)
{
	// Get the function
	const _function *ftn = tree->fptr;

	// Quit early if is a special instruction
	_node *diff_tree;
	if ((diff_tree = _diff_ispec(ftn, tree, vindex)))
		return diff_tree;

	// Otherwise, it (should) has a derivate overloaded
	_function *d = ftn->diff(vindex);

	// TODO: null check
	
	// TODO: what if d isnt iseq?
	if (d->spop == _function::op_iseq) {
		ISeq *diseq = reinterpret_cast <ISeq *> (d);

		ISeq::_cache_map cache;
		_node *dtree = diseq->_tree(cache);

		// std::cout << "PARTIALLY DIFFED TREE:\n" << dtree->str() << std::endl;

		_node *replaced = _diff_iseq(
			dtree,
			tree->children,
			diseq->_consts,
			vindex
		);

		// std::cout << "REPLACED:\n" << replaced->str() << std::endl;

		return replaced;
	}

	return nullptr;
}

// Differentiate the ISeq
_function *ISeq::diff(const int vindex) const
{
	// Get the tree
	_cache_map cache;
	_node *tree = _tree(cache);

	// std::cout << "PRE TREE:\n" << tree->str() << std::endl;

	// Differentiate the tree
	_node *diff_tree = _diff_tree(tree, vindex);

	// std::cout << "DIFF TREE:\n" << diff_tree->str() << std::endl;

	// Rebuild the instructions
	Instructions instrs;
	ConstantCache ccache;

	_rebuild(diff_tree, instrs, ccache, cache, _consts);

	// Return the function
	ISeq *diff_iseq = new ISeq(instrs, ccache, inputs);
	return diff_iseq;
}

}

}
