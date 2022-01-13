#ifndef ZHETAPI_FUNCTION_H_
#define ZHETAPI_FUNCTION_H_

// Standard headers
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
	// String versions of operations
	static constexpr const char *_spec_strs[] {
		"GET", "CONST",

		"VAR", "ISEQ",

		"ADD", "SUB",
		"MUL", "DIV"
	};
public:
	// Special operations
	enum Special {
		op_none = -1,

		op_get,
		op_const,

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

	// Constructors
	_function(int nins, int op = op_none) : inputs(nins), spop(op) {}

	// Not pure virtual so that special operations
	//	can get away without implementing it
	virtual Constant compute(const Input &) const {
		return Constant();
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

	// Process variadic arguments
	template <class ... Args>
	static void _process(_function::Input &input, Constant &c, Args ... args) {
		input.push_back(c);
		_process(input, args...);
	}

	static void _process(_function::Input &input, Constant &c) {
		input.push_back(c);
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
	template <class ...Args>
	Constant operator()(Args ...args) {
		_function::Input ins;
		_process(ins, args...);
		return fptr->compute(ins);
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

// Constant getter
struct Const : public _function {
	int index;

	Const(int i) : _function(1, op_const), index(i) {}
};

// Instruction sequence for a function
class ISeq : public _function {
	// Instructions are a sequence of functions
	std::vector <const _function *> _instrs;

	// Variables
	//	must be filled out before computation
	std::vector <_variable *> _vars;

	// Constants
	std::vector <Constant> _consts;

	// Load inputs to _vars
	void _load(const Input &ins) const {
		// TODO: should make sure that the ins == inputs
		for (int i = 0; i < ins.size(); i++)
			_vars[i]->value = ins[i];
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
		if (ops.size() < ftn->inputs) {}	// TODO: throw

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

		oss << "ISeq Instructions\n";
		for (int i = 0; i < _instrs.size(); i++) {
			oss << i << "\t" << _instrs[i]->summary()
				<< "\t" << _instrs[i] << std::endl;
		}

		return oss.str();
	}
};

// Standard ISeqs
#define binop(op, spop) 			\
	struct op : public ISeq {		\
		op() : ISeq ({			\
			new Get(0),		\
			new Get(1),		\
			new _function(2, spop)	\
		}, {}, 2) {}			\
	};

binop(Add, op_add)
binop(Sub, op_sub)
binop(Mul, op_mul)
binop(Div, op_div)

// Overloaded operators to create functions
Function operator+(const Function &, const Function &);
// Function operator-(const Variable &, const Variable &);

}

}

#endif
