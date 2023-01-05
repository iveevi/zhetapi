// Library headers
#include "../../include/autograd/iseq.hpp"
#include "../../include/common.hpp"

namespace zhetapi {

namespace autograd {

//////////////////////
// Public interface //
//////////////////////

// Empty constructor
ISeq::ISeq() : _function(0, op_iseq) {}

// Get variable
const ISeq::Var &ISeq::get(int index) const
{
	return _vars[index];
}

// Evaluate the sequence
Constant ISeq::compute(const Input &ins)
{
	// Load inputs
	_load(ins);

	// Stack of operands
	std::stack <Constant> ops;

	// Execute all the instructions
	for (const Ptr &ftn : _instrs)
		_exec(ftn, ops);

	// Return top-most value on the stack
	return ops.top();
}

// Evaluate gradient
_function::Gradient ISeq::gradient(const Input &ins, const Input &igrads)
{
	// Output gradient
	Gradient g;
	g.igrads = igrads;

	// TODO: possible extreme cache, check the input to previous input to
	// see if recomputation can be avoided
	// compute(ins);

	// Loop through all kernel functions
	// std::cout << "\nISeq gradient:\n";
	for (int i = _instrs.size() - 1; i >= 0; i--) {
		// Get function
		const Ptr &ftn = _instrs[i];

		// Get gradient
		if (_cached_finputs.count(ftn.get()) == 0)
			throw std::runtime_error("ISeq::gradient: cached inputs not found");
		
		Input ins = _cached_finputs[ftn.get()];

		/* std::cout << "function:\t" << ftn->summary() << std::endl;
		std::cout << "Input:" << std::endl;
		for (int j = 0; j < ins.size(); j++)
			std::cout << "\t" << ins[j] << std::endl; */

		_function::Gradient g_ftn = ftn->gradient(ins, g.igrads);

		// Add to gradients
		g.grads.insert(g.grads.end(), g_ftn.grads.begin(), g_ftn.grads.end());

		// Set input gradients
		g.igrads = g_ftn.igrads;
	}

	return g;
}

// Apply gradient
void ISeq::update_parameters(GradientQueue &grad_queue)
{
	// Loop through all kernel functions
	for (const Ptr &ftn : _instrs)
		ftn->update_parameters(grad_queue);
}

// Permute the order of variables
void ISeq::refactor(const std::vector <const _variable *> &vars)
{
	assert(vars.size() == _vars.size());

	// Generate indices of each variable
	_reindex_map reindex;
	for (int i = 0; i < _vars.size(); i++) {
		int j = index_of(vars[i]);
		assert(j >= 0 && reindex.count(j) == 0);
		reindex[j] = i;
	}

	// Reindex every Get instruction
	for (const Ptr &ftn : _instrs) {
		if (ftn->spop != op_get)
			continue;

		// Get variable index
		Get *get = reinterpret_cast <Get *> (ftn.get());
		int j = get->index;
		assert(reindex.count(j) > 0);
		get->index = reindex[j];
	}
}

_function::Ptr ISeq::refactor(const std::vector <const _variable *> &vars) const
{
	// Create copy
	ISeq *iseq = new ISeq(_instrs, _consts, inputs, _generate_reindex_map());
	iseq->refactor(vars);
	return Ptr(iseq);
}

// Info about parameters
int ISeq::parameters() const
{
	int n = 0;
	for (const Ptr &ftn : _instrs)
		n += ftn->parameters();
	return n;
}

int ISeq::tunable_parameters() const
{
	int n = 0;
	for (const Ptr &ftn : _instrs)
		n += ftn->tunable_parameters();
	return n;
}

// Dump instructions for debugging
std::string ISeq::summary() const
{
	// Headers
	io::Args args {
		"Index",
		"Instruction",
		"Variable (inputs=" + std::to_string(inputs) + ")",
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
ISeq::ISeq(const _function::Ptr &ftn, int nins)
		: _function(nins, op_iseq)
{
	// Fill the _vars with variables
	_vars.resize(nins);

	for (int i = 0; i < ftn->inputs; i++)
		_vars[i] = std::make_shared <_variable> ();

	// Add the get instructions
	_instrs.resize(nins + 1);

	int i = 0;
	for (; i < nins; i++)
		_instrs[i] = new_ftn_ <Get> (i);

	// Add the kernel
	_instrs[i] = ftn;
}

// List of instructions and constants
ISeq::ISeq(const std::vector <_function::Ptr> &instrs,
	std::vector <Constant> consts, int nins)
		: _function(nins, op_iseq),
		_instrs(instrs),
		_consts(consts)
{
	// Fill the _vars with variables
	_vars.resize(nins);

	for (int i = 0; i < nins; i++)
		_vars[i] = std::make_shared <_variable> ();
}

ISeq::ISeq(const std::vector <_function::Ptr> &instrs,
	std::vector <Constant> consts, int nins,
	const _reindex_map &reindex)
		: _function(nins, op_iseq),
		_instrs(instrs),
		_consts(consts)
{
	// Fill the _vars with variables
	_vars.resize(nins);

	for (int i = 0; i < nins; i++)
		_vars[i] = std::make_shared <_variable> (reindex.at(i));
}

std::pair <_function *, const _function::MethodTable &> ISeq::method_table()
{
	// Choose the final instruction to request method table from
	for (int i = _instrs.size() - 1; i >= 0; i--) {
		if (_instrs[i]->spop != op_get)
			return _instrs[i]->method_table();
	}

	// ISeq properties?
	static const MethodTable _map {};
	return {this, _map};
}

//////////////////////
// Append functions //
//////////////////////

// Append helpers
void ISeq::_append_function(const _function::Ptr &fptr)
{
	int index;

	switch (fptr->spop) {
	case op_get:
		// If there are fewer inputs that requested,
		// then add new variables
		index = reinterpret_cast <const Get *> (fptr.get())->index;
		for (int i = inputs; i <= index; i++)
			_vars.push_back(_variable::new_var());
		_instrs.push_back(fptr);
		inputs = _vars.size();
		break;
	case op_var:
		append_variable((_variable *) fptr.get());
		break;
	case op_iseq:
		append_iseq((ISeq *) fptr.get());
		break;
	default:
		// Just add the function to the instructions
		_instrs.push_back(fptr);
		break;
	}
}

void ISeq::append_variable(const _variable *v)
{
	// Check if the variable exists already
	int index = index_of(v);

	// Add the variable (and increment inputs) if not found
	int gindex = index;
	if (index < 0) {
		gindex = _vars.size();
		_vars.push_back(v->clone());
		inputs++;
	}

	_instrs.push_back(new_ftn_ <Get> (gindex));
}

void ISeq::append_iseq(const ISeq *const iseq)
{
	for (const Ptr &fptr : iseq->_instrs) {
		// TODO: avoid copies
		Ptr nptr = fptr;
		if (nptr->spop == op_get) {
			// TODO: clean up
			int i = reinterpret_cast <Get *> (nptr.get())->index;
			const Var &v = iseq->_vars[i];

			// TODO: should create a new value as well
			i = index_of(v.get());

			if (i == -1) {
				// Add a variablei
				i = _vars.size();
				_vars.push_back(v->clone());
				inputs++;
			}

			_append_function(new_ftn_ <Get> (i));
			continue;
		} else if (nptr->spop == op_const) {
			int index = reinterpret_cast <const Const *> (nptr.get())->index;
			Constant c = iseq->_consts[index];

			// Add the constant to this ISeq
			_consts.push_back(c);
			_append_function(new_ftn_ <Const> (_consts.size() - 1));
		}

		_append_function(nptr);
	}
}

// Get index of variable
int ISeq::index_of(const _variable *v) const
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
static Constant _get_operand(std::stack <Constant> &ops, _function::Input &ins)
{
	// Check empty first
	if (ops.empty())
		throw std::runtime_error("ISeq::_get_operand: empty stack");

	Constant c = ops.top();
	ins.push_back(c);
	ops.pop();

	return c;
}

// Operation kernels
using Kernel = Constant (*)(std::stack <Constant> &, _function::Input &);

// List of kernels
static std::vector <Kernel> kernels = {
	// Addition
	[](std::stack <Constant> &ops, _function::Input &ins) -> Constant {
		Constant b = _get_operand(ops, ins);
		Constant a = _get_operand(ops, ins);
		return a + b;
	},

	// Subtraction
	[](std::stack <Constant> &ops, _function::Input &ins) -> Constant {
		Constant b = _get_operand(ops, ins);
		Constant a = _get_operand(ops, ins);
		return a - b;
	},

	// Multiplication
	[](std::stack <Constant> &ops, _function::Input &ins) -> Constant {
		Constant b = _get_operand(ops, ins);
		Constant a = _get_operand(ops, ins);
		return a * b;
	},

	// Division
	[](std::stack <Constant> &ops, _function::Input &ins) -> Constant {
		Constant b = _get_operand(ops, ins);
		Constant a = _get_operand(ops, ins);
		return a / b;
	}
};

// Load inputs
void ISeq::_load(const Input &ins)
{
	// Ensure correct number of inputs
	if (ins.size() != inputs) {
		std::stringstream ss;
		ss << "ISeq::_load: expected " << inputs << " inputs, got " << ins.size();
		std::cout << "vars:" << std::endl;
		for (const Var &v : _vars) {
			std::cout << "\t" << v->summary()
				<< " (" << v->id << ")" << std::endl;
		}
		throw std::runtime_error(ss.str());
	}

	_cached_in = ins;
	for (int i = 0; i < ins.size(); i++)
		_vars[i]->value = ins[i];
}

// Storing a constant into the cache
void ISeq::storec(std::stack <Constant> &ops, int i) const
{
	// Get from the stack
	// TODO: should the input for this be cached?
	_function::Input ins;
	Constant c = _get_operand(ops, ins);

	if (i >= _cache.size())
		_cache.resize(i + 1);

	_cache[i] = c;
}

// Deal with special instructions
bool ISeq::_ispec(const Ptr &ftn, std::stack <Constant> &ops)
{
	_function::Input ins;
	int index;

	// TODO: be more conservative with memory...
	switch (ftn->spop) {
	case op_get:
		// Get index and push the corresponding variable
		index = reinterpret_cast <const Get *> (ftn.get())->index;
		_cached_finputs[ftn.get()] = Input {_vars[index]->value.copy()};
		ops.push(_vars[index]->value.copy());
		return true;
	case op_const:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const Const *> (ftn.get())->index;
		ops.push(_consts[index].copy());
		return true;
	case op_store_cache:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const _store_cache *> (ftn.get())->index;
		storec(ops, index);
		return true;
	case op_get_cache:
		// Get index and push the corresponding constant
		index = reinterpret_cast <const _get_cache *> (ftn.get())->index;
		ops.push(_cache[index].copy());
		return true;
	case op_add: case op_sub:
	case op_mul: case op_div:
		// Push the result of the operation
		ops.push(kernels[ftn->spop - op_add](ops, ins).copy());
		_cached_finputs[ftn.get()] = ins;
		return true;
	default:
		break;
	}

	return false;
}

// Execute an instruction
void ISeq::_exec(const Ptr &ftn, std::stack <Constant> &ops)
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

	// Store input in cache
	// TODO: this many copies may be expensive
	Input copy;
	for (const Constant &c : fins)
		copy.push_back(c.copy());
	_cached_finputs[ftn.get()] = copy;

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
		const _function::Ptr &ftn,
		std::vector <Constant> &dst_consts,
		const std::vector <Constant> &src_consts,
		std::unordered_map <int, int> &reindex,
		int &vindex)
{
	// If the instruction is a get,
	// add the corresponding variable
	if (ftn->spop == _function::op_get) {
		int index = reinterpret_cast <const Get *> (ftn.get())->index;

		// Get variable id from iseq
		int id = iseq->get(index)->id;

		// Check if the variable is already in reindex map
		if (reindex.find(id) == reindex.end()) {
			// Add the variable to the reindex map
			reindex[id] = vindex;

			// Add the variable to the instruction sequence
			instrs.push_back(new_ftn_ <Get> (vindex));

			// Increment the variable index
			vindex++;
		} else {
			// Add the variable to the instruction sequence
			instrs.push_back(new_ftn_ <Get> (reindex[id]));
		}
	} else if (ftn->spop == _function::op_const) {
		// TODO: need to lookup if the constant is already in the cache
		int index = reinterpret_cast <const Const *> (ftn.get())->index;

		// Add the constant to the instruction sequence
		// For now, duplicate consts are fine
		instrs.push_back(new_ftn_ <Const> (dst_consts.size()));
		dst_consts.push_back(src_consts[index]);
	} else {
		instrs.push_back(ftn);
	}
}

// Override composition
_function::Ptr ISeq::_compose(const Compositions &cs) const
{
	/* std::cout << "Composing: " << std::endl;
	for (const auto &c : cs)
		std::cout << c->summary() << std::endl; */

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
		const _function::Ptr &ftn = cs[i];

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
			id = reinterpret_cast <const _variable *> (ftn.get())->id;
			reindex[id] = vindex;
			instrs.push_back(new_ftn_ <Get> (vindex++));
			break;
		case op_iseq:
			// Get the iseq, and append its instructions
			iseq = reinterpret_cast <const ISeq *> (ftn.get());

			for (const Ptr &fptr : iseq->_instrs)
				_compose_iseq(instrs, iseq, fptr, consts, iseq->_consts, reindex, vindex);
			break;
		case op_repl_const:
			rc = reinterpret_cast <const _repl_const *> (ftn.get());
			consts.push_back(rc->value);
			instrs.push_back(new_ftn_ <Const> (rc->index));
			break;
		default:
			// Add the instruction
			instrs.push_back(ftn);
			break;
		}

		// Add store cache instruction
		instrs.push_back(new_ftn_ <_store_cache> (i));
	}

	// Add own instructions
	for (const Ptr &ftn : _instrs) {
		int index;

		switch (ftn->spop) {
		case op_get:
			// Changes to get cache
			index = reinterpret_cast <const Get *> (ftn.get())->index;
			instrs.push_back(new_ftn_ <_get_cache> (index));
			break;
		default:
			// Add the instruction
			instrs.push_back(ftn);
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
	return Ptr(iseq);
}

//////////////////////////////////
// Tree building and rebuilding //
//////////////////////////////////

// Constructors for the tree
_node::_node(const _function::Ptr &f) : fptr(f) {}
_node::_node(const _function::Ptr &f, const std::vector <Node> &cs)
		: fptr(f), children(cs) {}

Node _node::make(const _function::Ptr &f)
{
	return Node(new _node(f));
}

Node _node::make(const _function::Ptr &f, const std::vector <Node> &cs)
{
	return Node(new _node(f, cs));
}

// Get string
std::string _node::str(int indent) const
{
	// Get string of current node
	std::string s = fptr->summary();

	// Indentation
	std::string sindent(indent, '\t');

	// If there are children
	for (const Node &child : children) {
		s += "\n" + sindent + "\u2514\u2500\u2500 "
			+ child->str(indent + 1);
	}

	return s;
}

// Constructors for cache info
ISeq::_cache_info::_cache_info() {}
ISeq::_cache_info::_cache_info(int r, const Node &tree)
		: refs(r), value(tree) {}

// Generating the tree
Node ISeq::_tree(_cache_map &cache) const
{
	// Keep a stack of nodes,
	// 	similar to computation
	std::stack <Node> nodes;

	// Iterate through the instructions
	for (const Ptr &ftn : _instrs)
		_tree_walk(ftn, nodes, cache);

	// Return the top node and free the stack
	Node top = nodes.top();
	nodes.pop();

	return top;
}

// TODO: static method?
void ISeq::_tree_walk(const Ptr &ftn, std::stack <Node> &nodes, _cache_map &cache) const
{
	// Switch variables
	std::vector <Node> children;
	Node value;
	int index;

	// Check type of operation
	switch (ftn->spop) {
	case op_get:
		break;
	case op_get_cache:
		// Get index
		index = reinterpret_cast <const _get_cache *> (ftn.get())->index;

		// Get value from cache map
		value = cache[index].value;

		// Increment the reference count
		cache[index].refs++;

		children = {value};
		break;
	case op_store_cache:
		// Get index of cache
		index = reinterpret_cast <const _get_cache *> (ftn.get())->index;

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
			Node child = nodes.top();
			nodes.pop();

			// Push the child (to the front)
			children.insert(children.begin(), child);
		}

		// Add the node
		break;
	}

	// Add the node
	nodes.push(_node::make(ftn, children));
}

// Rebuild the ISeq from the tree
void ISeq::_rebuild(const Node &tree, Instructions &instrs,
		ConstantCache &ccache, _cache_map &cache,
		const ConstantCache &pconsts) const
{
	// Iterate through the tree
	const _function::Ptr &ftn = tree->fptr;

	if (ftn->spop == op_get_cache) {
		// Get the index
		int index = reinterpret_cast
			<const _get_cache *> (ftn.get())->index;

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
			<const _repl_const *> (ftn.get())->value;

		// TODO: should optimize constants if there are a lot

		// Add the constant
		int index = ccache.size();
		ccache.push_back(c);

		// Add instruction
		instrs.push_back(new_ftn_ <Const> (index));
	} else if (ftn->spop == op_const) {
		// std::cout << "Constant index = " << reinterpret_cast <const Const *> (ftn)->index << std::endl;
		// std::cout << "\tpconsts size = " << pconsts.size() << std::endl;
		Constant c = pconsts[reinterpret_cast
			<const Const *> (ftn.get())->index
		];

		// Add the constant
		int index = ccache.size();
		ccache.push_back(c);

		// Add instruction
		instrs.push_back(new_ftn_ <Const> (index));
	} else {
		// Add the operands
		for (const Node &child : tree->children)
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
	Node tree = _tree(cache);

	/* std::cout << "\nTree: " << std::endl;
	std::cout << tree->str() << std::endl; */

	Instructions instrs;
	ConstantCache ccache;
	_rebuild(tree, instrs, ccache, cache, _consts);

	// Replace the instructions and constant cache
	// TODO: clean up excess memory
	_instrs = instrs;
	_consts = ccache;

	/* TODO: prune duplicate constants
	std::cout << "Optimized ISeq:" << std::endl;
	std::cout << summary() << std::endl;

	std::cout << "\nconsts: " << _consts.size() << std::endl;
	for (int i = 0; i < _consts.size(); i++)
		std::cout << "\t" << i << ": " << _consts[i] << std::endl;
	std::cout << "===========================" << std::endl; */
}

// Generate reindex map
ISeq::_reindex_map ISeq::_generate_reindex_map() const
{
	_reindex_map reindex;
	for (int i = 0; i < _vars.size(); i++)
		reindex[i] = _vars[i]->id;
	return reindex;
}

/////////////////////
// Differentiation //
/////////////////////

// Forward declarations
Node _diff_tree(const Node &, int);

// Differentiation kernels
Node _diffk_get(const _function::Ptr &fptr, const Node &tree, int vindex)
{
	// Get the index
	int index = reinterpret_cast <const Get *> (fptr.get())->index;

	// Check if the index is the same as the variable
	if (index == vindex)
		return _node::make(new_ftn_ <_repl_const> (1, vindex));

	// Add the node
	return _node::make(new_ftn_ <_repl_const> (0, vindex));
}

Node _diffk_add_sub(const _function::Ptr &fptr, const Node &tree, int vindex)
{
	_function::Ptr f = new_ftn_ <_function> (1, fptr->spop);
	Node d1 = _diff_tree(tree->children[0], vindex);
	Node d2 = _diff_tree(tree->children[1], vindex);
	return _node::make(f, {d1, d2});
}

Node _diffk_mul(const _function::Ptr &fptr, const Node &tree, int vindex)
{
	_function::Ptr f1 = new_ftn_ <_function> (1, _function::op_add);
	_function::Ptr f2 = new_ftn_ <_function> (1, _function::op_mul);

	Node c1 = tree->children[0];
	Node c2 = tree->children[1];

	Node d1 = _diff_tree(c1, vindex);
	Node d2 = _diff_tree(c2, vindex);

	Node c3 = _node::make(f2, {c1, d2});
	Node c4 = _node::make(f2, {c2, d1});

	return _node::make(f1, {c4, c3});
}

Node _diffk_div(const _function::Ptr &fptr, const Node &tree, int vindex)
{
	_function::Ptr f1 = new_ftn_ <_function> (1, _function::op_sub);
	_function::Ptr f2 = new_ftn_ <_function> (1, _function::op_mul);
	_function::Ptr f3 = new_ftn_ <_function> (1, _function::op_div);

	Node c1 = tree->children[0];
	Node c2 = tree->children[1];

	Node d1 = _diff_tree(c1, vindex);
	Node d2 = _diff_tree(c2, vindex);

	Node c3 = _node::make(f2, {c2, d1});
	Node c4 = _node::make(f2, {c1, d2});

	Node c5 = _node::make(f1, {c3, c4});
	Node c6 = _node::make(f2, {c2, c2});

	return _node::make(f3, {c5, c6});
}

// Construct tree for special instructions
Node _diff_ispec(const _function::Ptr &ftn,
		const Node &tree, int vindex)
{
	// TODO: operation factories (which return shared_ptr)

	// TODO: map for kernels?

	// Check type of operation
	switch (ftn->spop) {
	case _function::op_get:
		return _diffk_get(ftn, tree, vindex);
	case _function::op_const:
		return _node::make(new_ftn_ <_repl_const> (0, vindex));
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
Node _diff_iseq(const Node &orig,
		const std::vector <Node> &ins,
		const std::vector <Constant> &consts,
		int vindex)
{
	// TODO: check that there are no more ISeqs
	_function::Ptr ftn = orig->fptr;

	// Temporary indices
	int index = -1;

	// Any new children
	std::vector <Node> children;

	// Switch special instructions
	switch (ftn->spop) {
	case _function::op_get:
		// Get the index
		index = reinterpret_cast <const Get *> (ftn.get())->index;
		return ins[index];
	case _function::op_const:
		// Replace with _repl_const
		index = reinterpret_cast <const Const *> (ftn.get())->index;
		return _node::make(new_ftn_ <_repl_const> (consts[index], -1));
	case _function::op_differential:
		// Get the index
		index = reinterpret_cast <const _iop *> (ftn.get())->index;
		return _diff_tree(ins[index], vindex);
	default:
		// Reiterate children
		for (const Node &child : orig->children)
			children.push_back(_diff_iseq(child, ins, consts, vindex));

		// Create new node
		return _node::make(ftn, children);
	}

	return nullptr;
}

// Construct a new, differentiated tree
Node _diff_tree(const Node &tree, int vindex)
{
	// Get the function
	const _function::Ptr &ftn = tree->fptr;

	// Quit early if is a special instruction
	Node diff_tree;
	if ((diff_tree = _diff_ispec(ftn, tree, vindex)))
		return diff_tree;

	// Otherwise, it (should) has a derivate overloaded
	_function::Ptr d = ftn->diff(vindex);

	// TODO: null check

	// TODO: what if d isnt iseq?
	if (d->spop == _function::op_iseq) {
		ISeq *diseq = reinterpret_cast <ISeq *> (d.get());

		ISeq::_cache_map cache;
		Node dtree = diseq->_tree(cache);

		// std::cout << "PARTIALLY DIFFED TREE:\n" << dtree->str() << std::endl;

		Node replaced = _diff_iseq(
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
_function::Ptr ISeq::diff(const int vindex) const
{
	// Get the tree
	_cache_map cache;
	Node tree = _tree(cache);

	// std::cout << "PRE TREE:\n" << tree->str() << std::endl;

	// Differentiate the tree
	Node diff_tree = _diff_tree(tree, vindex);

	// std::cout << "DIFF TREE:\n" << diff_tree->str() << std::endl;

	// Rebuild the instructions
	Instructions instrs;
	ConstantCache ccache;

	_rebuild(diff_tree, instrs, ccache, cache, _consts);

	// Return the function
	return Ptr(new ISeq(instrs, ccache, inputs));
}

}

}
