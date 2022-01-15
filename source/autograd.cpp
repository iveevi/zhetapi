#include "../include/autograd/autograd.hpp"

// Macros
#define KERNEL(name)						\
	Function name = new_ <_##name> ();			\
	inline Constant _k##name(const _function::Input &ins)

namespace zhetapi {

namespace autograd {

// ISeq functions
void ISeq::_load(const Input &ins) const
{
	// TODO: should make sure that the ins == inputs
	for (int i = 0; i < ins.size(); i++)
		_vars[i]->value = ins[i];
}

// Get constant from the stack and handle any errors
Constant ISeq::getc(std::stack <Constant> &ops)
{
	// TODO: check empty
	Constant c = ops.top();
	ops.pop();

	return c;
}

// Storing a constant into the cache
void ISeq::storec(std::stack <Constant> &ops, int i) const
{
	// Get from the stack
	Constant c = getc(ops);

	if (i >= _cache.size())
		_cache.resize(i + 1);

	_cache[i] = c;
}

// Deal with special instructions
bool ISeq::_ispec(const _function *ftn, std::stack <Constant> &ops) const
{
	// Operation kernels (for math)
	using Kernel = Constant (*)(std::stack <Constant> &);

	// List of kernels
	static std::vector <Kernel> kernels = {
		[](std::stack <Constant> &ops) -> Constant {
			Constant b = ops.top();
			ops.pop();
			Constant a = ops.top();
			ops.pop();
			return a + b;
		},

		[](std::stack <Constant> &ops) -> Constant {
			Constant b = ops.top();
			ops.pop();
			Constant a = ops.top();
			ops.pop();
			return a - b;
		},

		[](std::stack <Constant> &ops) -> Constant {
			Constant b = getc(ops);
			Constant a = getc(ops);

			// TODO: use matrix multiplication if dims == 2
			// or if dim== 2 and dim == 1 -> user must do hammard
			// for element-wise multiplication
			return multiply(a, b);
		},

		[](std::stack <Constant> &ops) -> Constant {
			Constant b = getc(ops);
			Constant a = getc(ops);
			return divide(a, b);
		}
	};

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

// Protected constructors
ISeq::ISeq(std::vector <const _function *> instrs,
	std::vector <Constant> consts, int nins)
		: _function(nins, op_iseq),
		_instrs(instrs),
		_consts(consts)
{
	// Fill the _vars with variables
	_vars.resize(nins);
}

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

// Overload composition
_function *ISeq::_compose(const Compositions &cs) const
{
	// Composition returns a new ISeq
	ISeq *iseq = new ISeq();

	// Append all the _functions
	int i = 0;
	for (const _function *fptr : cs) {
		// Get if the fptr is repl const
		if (fptr->spop == op_repl_const) {
			// TODO: method
			const _repl_const *rc = reinterpret_cast <const _repl_const *> (fptr);

			int ci = iseq->_consts.size();
			iseq->_consts.push_back(rc->value);
			iseq->_instrs.push_back(new Const(ci));
			iseq->_instrs.push_back(new _store_cache(i++));
			continue;
		}

		iseq->append(fptr);

		// Then store into cache
		iseq->append(new _store_cache(i++));

		// TODO: use append_scache -> int cache index
		// when the cache is used for other things
	}

	// Add self
	// TODO: need to strip all of the gets,
	// and replace with the fptrs
	// iseq->append_iseq((ISeq *) this);

	// Iterate through our instructions
	//	and replace get with get_cache
	for (const _function *fptr : _instrs) {
		if (fptr->spop == op_get) {
			i = reinterpret_cast <const Get *> (fptr)->index;
			iseq->_instrs.push_back(new _get_cache(i));
			continue;
		}

		iseq->append(fptr);
	}

	// TODO: later optimize the sequence store-cache(i), get-cache(i)

	// Return the sequence
	return iseq;
}
	
// Empty constructor
ISeq::ISeq() : _function(0, op_iseq) {}

// Inserting instructions and functions
void ISeq::append(const _function *fptr) {
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

// Append a sequence of instructions
template <class ... Args>
void ISeq::append(const _function *fptr, Args ... args)
{
	append(fptr);
	append(args...);
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

// Static variables
constexpr const char *_function::_spec_strs[];

// Operators
Function operator+(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new _function(2, _function::op_add)
	);

	return Function(iseq);
}

Function operator-(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new _function(2, _function::op_sub)
	);

	return Function(iseq);
}

Function operator*(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new _function(2, _function::op_mul)
	);

	return Function(iseq);
}

Function operator/(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new _function(2, _function::op_div)
	);

	return Function(iseq);
}

// Standard function kernels
KERNEL(sqrt)
{
	return ins[0].transform(sqrtl);
}

KERNEL(exp)
{
	return ins[0].transform(expl);
}

KERNEL(log)
{
	return ins[0].transform(logl);
}

KERNEL(sin)
{
	return ins[0].transform(sinf);
}

KERNEL(cos)
{
	return ins[0].transform(cosf);
}

KERNEL(tan)
{
	return ins[0].transform(tanf);
}

KERNEL(pow)
{
	// Use only the first element
	long double e = ins[1].get(0);

	return ins[0].transform(
		[e](long double x) -> long double {
			return powl(x, e);
		}
	);
}

}

}
