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

		// TODO: should print shape, not value
		if (i < _consts.size())
			row.push_back("--shape--");
		else
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
void ISeq::append_variable(_variable *v)
{
	// Check if the variable exists already
	int index = index_of(v);

	// Add the variable (and increment inputs) if not found
	int gindex = index;
	if (index < 0) {
		gindex = _vars.size();
		_vars.push_back(v);
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
			}

			append(new Get (i));
			continue;
		}

		append(nptr);
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
	// TODO: should make sure that the ins == inputs
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
		ops.push(_vars[index]->value);
		return true;
	case op_const:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const Const *> (ftn)->index;
		ops.push(_consts[index]);
		return true;
	case op_store_cache:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const _store_cache *> (ftn)->index;
		storec(ops, index);
		return true;
	case op_get_cache:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const _get_cache *> (ftn)->index;
		ops.push(_cache[index]);
		return true;
	case op_add: case op_sub:
	case op_mul: case op_div:
		// Push the result of the operation
		ops.push(kernels[ftn->spop - op_add](ops));
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
	if (_ispec(ftn, ops))
		return;

	// Ensure that the stack has enough operands
	if (ops.size() < ftn->inputs) {}	// TODO: throw, except if #inputs = -1 (variadic)

	// Load the inputs
	Input fins;
	for (int i = 0; i < ftn->inputs; i++) {
		Constant c = ops.top();
		ops.pop();

		fins.push_back(c);
	}

	// Evaluate the instruction
	Constant n = ftn->compute(fins);

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

		const ISeq *iseq;
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

	// std::cout << "===New ISeq===\n" << iseq->summary() << std::endl;

	// Optimize the ISeq
	iseq->_optimize();

	// std::cout << "\n===Final ISeq===\n" << iseq->summary() << std::endl;

	return iseq;
}

/////////////////////////////////////////
// Optimizing the instruction sequence //
/////////////////////////////////////////

// Optimize the ISeq
void ISeq::_optimize()
{
	// std::cout << "OPTIMIZING ISEQ" << std::endl;

	_cache_map cache;
	_node *tree = _tree(cache);

	std::cout << "TREE:\n" << tree->str() << std::endl;

	Instructions instrs;
	_rebuild(tree, instrs, cache);

	/* std::cout << "INSTRUCS:\n" << std::endl;
	for (const _function *fptr : instrs)
		std::cout << fptr->summary() << std::endl; */

	// Replace the instructions
	// TODO: clean up excess memory
	_instrs = instrs;
}

// Constructors for the tree
ISeq::_node::_node(const _function *f) : fptr(f) {}
ISeq::_node::_node(const _function *f, const std::vector <_node *> &cs)
		: fptr(f), children(cs) {}

// Get string
std::string ISeq::_node::str(int indent) const
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
ISeq::_node *ISeq::_tree(_cache_map &cache) const
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
void ISeq::_rebuild(const _node *tree, Instructions &instrs, _cache_map &cache) const
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
			_rebuild(info.value, instrs, cache);
		} else {
			// Add get cache
			instrs.push_back(ftn);
		}
	} else {
		// Add the operands
		for (const _node *child : tree->children)
			_rebuild(child, instrs, cache);

		// Add the instruction
		instrs.push_back(ftn);
	}
}

}

}
