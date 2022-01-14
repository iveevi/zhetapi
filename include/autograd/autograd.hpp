#ifndef ZHETAPI_FUNCTION_H_
#define ZHETAPI_FUNCTION_H_

// Standard headers
#include <cmath>
#include <memory>
#include <sstream>
#include <stack>
#include <vector>

// Library headers
#include "../tensor.hpp"

namespace zhetapi {

namespace autograd {

// Constants are just tensors
using Constant = Tensor <long double>;

// Basic structure of a function
class _function {
public:
	// Special operations
	enum Special {
		op_none = -1,

		// Key ISeq operations
		op_get,
		op_const,
		op_store_cache,
		op_get_cache,

		op_var,
		op_iseq,

		// Basic math operations
		op_add,
		op_sub,
		op_mul,
		op_div
	};

	// Special operation code (defualt -1 = none)
	int spop = op_none;

	int inputs = -1;

	// Type of input
	using Input = std::vector <Constant>;
	using Compositions = std::vector <_function *>;
private:
	// String versions of operations
	static constexpr const char *_spec_strs[] {
		"GET", "CONST",
		"STORE CACHE", "GET CACHE",

		"VAR", "ISEQ",

		"ADD", "SUB",
		"MUL", "DIV"
	};
protected:
	// By default, function composition returns null
	virtual _function *_compose(const Compositions &) const {
		return nullptr;
	}
public:
	// Constructors
	_function(int nins, int op = op_none) : inputs(nins), spop(op) {}

	// Not pure virtual so that special operations
	//	can get away without implementing it
	virtual Constant compute(const Input &) const {
		// TODO: separate into _compute, like compose?
		return Constant();
	}

	// Wrapper around _compose that checks #inputs
	_function *compose(const Compositions &cs) const {
		// TODO: add argsize exception
		if (cs.size() != inputs)
			throw "_function::compose size mismatch";

		return this->_compose(cs);
	}

	// Copy pointer
	virtual _function *copy() const {
		return new _function(inputs, spop);
	}

	// Summary for each function
	virtual std::string summary() const {
		if (spop > op_none)
			return _spec_strs[spop];

		return "_function(#inputs = "
			+ std::to_string(inputs) + ")";
	}
};

// Function alias is a wrapper around shared ptr
using _fptr = std::shared_ptr <_function>;

class Function {
	// Function pointer
	_fptr fptr;

	// Process variadic arguments for compute
	template <class ... Args>
	static void _cmp_process(_function::Input &input, const Constant &c, Args ... args) {
		input.push_back(c);
		_cmp_process(input, args...);
	}

	template <class ... Args>
	static void _cmp_process(_function::Input &input, const Constant::value_type &c, Args ... args) {
		input.push_back(c);
		_cmp_process(input, args...);
	}

	static void _cmp_process(_function::Input &input, const Constant &c) {
		input.push_back(c);
	}

	static void _cmp_process(_function::Input &input, const Constant::value_type &c) {
		input.push_back(c);
	}

	// Process variadic arguments for compose
	template <class ... Args>
	static void _ftr_process(_function::Compositions &cs, const Function &f, Args ... args) {
		cs.push_back(f.get());
		_ftr_process(cs, args...);
	}

	static void _ftr_process(_function::Compositions &cs, const Function &f) {
		cs.push_back(f.get());
	}
public:
	// Constructors
	Function() : fptr(nullptr) {}
	Function(_function *f) : fptr(f) {}
	Function(const _fptr &f) : fptr(f) {}

	// Get raw handle
	_function *get() const {
		return fptr.get();
	}

	// Operator overloading
	Constant operator()(const std::vector <Constant> &ins) {
		return fptr->compute(ins);
	}

	// Variadic operator overloading
	//	needs at least one argument
	template <class ...Args>
	Constant operator()(const Constant &c, Args ...args) {
		_function::Input ins;
		_cmp_process(ins, c, args...);
		return fptr->compute(ins);
	}

	// Variadic operator overloading
	//	for composition
	template <class ...Args>
	Function operator()(const Function &f, Args ...args) {
		_function::Compositions cs;
		_ftr_process(cs, f, args...);
		return Function(fptr->compose(cs));
	}

	// Summary of functiooon
	std::string summary() const {
		return fptr->summary();
	}
};

// Allocate function
template <class T, class... Args>
Function new_(Args ... args)
{
	_function *f = new T(args...);
	return Function(_fptr(f));
}

// Get function
// TODO: underscore convention?
struct Get : public _function {
	int index;

	Get(int i) : _function(1, op_get), index(i) {}

	_function *copy() const override {
		return new Get(index);
	}

	// Overload summary to include index
	std::string summary() const override {
		return "GET (" + std::to_string(index) + ")";
	}
};

// Constant getter
struct Const : public _function {
	int index;

	Const(int i) : _function(1, op_const), index(i) {}
};

// Store into cache
// TODO: make a overview class -> index_spop(spop, index)
struct _store_cache : public _function {
	int index;

	_store_cache(int i) : _function(1, op_store_cache), index(i) {}

	_function *copy() const override {
		return new _store_cache(index);
	}

	// Overload summary to include index
	std::string summary() const override {
		return "STORE-CACHE (" + std::to_string(index) + ")";
	}
};

// Get from cache
struct _get_cache : public _function {
	int index;

	_get_cache(int i) : _function(1, op_get_cache), index(i) {}

	_function *copy() const override {
		return new _get_cache(index);
	}

	// Overload summary to include index
	std::string summary() const override {
		return "GET-CACHE (" + std::to_string(index) + ")";
	}
};

// Variables are just placeholders
class _variable : public _function {
	static int gid() {
		static int cid = 0;
		return cid++;
	}

	_variable(int x, const Constant &c)
		: _function(0, op_var), id(x), value(c) {}
public:
	// Unique variable id
	int id;

	// Value of the variable
	Constant value;

	_variable() : _function(0, op_var), id(gid()) {}

	_function *copy() const override {
		return new _variable(id, value);
	}
};

// Wrapper around _variable for convenience
class Variable : public Function {
public:
	// Constructors
	Variable() : Function(new_ <_variable> ()) {}
};

// Instruction sequence for a function
class ISeq : public _function {
	// TODO: JIT function to compile into object code
	//	this should be possible since the types
	//	are homogenous

	// Instructions are a sequence of functions
	std::vector <const _function *>	_instrs;

	// Variables
	//	must be filled out before computation
	std::vector <_variable *>	_vars;

	// Constants
	std::vector <Constant>		_consts;

	// Cache of constants
	mutable std::vector <Constant>	_cache;

	// Load inputs to _vars
	void _load(const Input &ins) const {
		// TODO: should make sure that the ins == inputs
		for (int i = 0; i < ins.size(); i++)
			_vars[i]->value = ins[i];
	}

	// Get constant from the stack and handle any errors
	static Constant getc(std::stack <Constant> &ops) {
		// TODO: check empty
		Constant c = ops.top();
		ops.pop();

		return c;
	}

	// Storing a constant into the cache
	void storec(std::stack <Constant> &ops, int i) const {
		// Get from the stack
		Constant c = getc(ops);

		if (i >= _cache.size())
			_cache.resize(i + 1);

		_cache[i] = c;
	}

	// Deal with special instructions
	bool _ispec(const _function *ftn, std::stack <Constant> &ops) const {
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
	// TODO: static?
	void _exec(const _function *ftn, std::stack <Constant> &ops) const {
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
	int index_of(_variable *v) {
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
	void append_variable(_variable *v) {
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

	void append_iseq(ISeq *iseq) {
		for (const _function *fptr : iseq->_instrs) {
			_function *nptr = fptr->copy();
			if (nptr->spop == op_get) {
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
protected:
	// Protected constructor
	ISeq(std::vector <const _function *> instrs,
		std::vector <Constant> consts,
		int nins) : _function(nins, op_iseq),
			_instrs(instrs),
			_consts(consts) {
		// Fill the _vars with variables
		_vars.resize(nins);
	}

	// Overload composition
	_function *_compose(const Compositions &cs) const override {
		// Composition returns a new ISeq
		ISeq *iseq = new ISeq();

		// Append all the _functions
		int i = 0;
		for (const _function *fptr : cs) {
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
public:
	// TODO: check function to make sure only
	// one element remains on the stack

	// Empty constructor
	ISeq() : _function(0, op_iseq) {}

	// Inserting instructions and functions
	void append(const _function *fptr) {
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
	void append(const _function *fptr, Args ... args) {
		append(fptr);
		append(args...);
	}

	// Evaluate the sequence
	Constant compute(const Input &ins) const {
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

	_function *copy() const override {
		return new ISeq(_instrs, _consts, inputs);
	}

	// Dump instructions for debugging
	std::string summary() const override {
		std::ostringstream oss;

		oss << "\nISeq Instructions\n";
		for (int i = 0; i < _instrs.size(); i++) {
			oss << i << "\t" << _instrs[i]->summary()
				<< "\t" << _instrs[i] << std::endl;
		}

		return oss.str();
	}
};

// Overloaded operators
Function operator+(const Function &, const Function &);
Function operator-(const Function &, const Function &);
Function operator*(const Function &, const Function &);
Function operator/(const Function &, const Function &);

// Specialized function classes
// TODO: separate header
// TODO: macros
class _sqrt : public ISeq {
	// Function kernel
	struct kernel : public _function {
		kernel() : _function(1) {}

		// Evaluation is element-wise square root
		Constant compute(const Input &ins) const override {
			return ins[0].transform(
				[](long double x) -> long double {
					return std::sqrt(x);
				}
			);
		}

		// Summary returns same thing as the kernel summary
		std::string summary() const override {
			return "SQRT";
		}
	};
public:
	_sqrt() : ISeq({
			new Get(0),
			new kernel()
		}, {}, 1) {}
};

// Function classes objects
//	all should be ISeqs, so that function composition
//	works as expected
extern Function sqrt;

}

}

#endif
