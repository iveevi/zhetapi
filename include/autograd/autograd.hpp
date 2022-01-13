#ifndef ZHETAPI_FUNCTION_H_
#define ZHETAPI_FUNCTION_H_

// Standard headers
#include <memory>
#include <stack>
#include <vector>

// Library headers
#include "../tensor.hpp"

namespace zhetapi {

namespace autograd {

// Constants are just tensors
using Constant = Tensor <long double>;

// Basic structure of a function
struct _function {
	// Special operations
	enum Special {
		op_none = -1,
		op_get,
		op_const,

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
	virtual Constant compute(const Input &) {
		return Constant();
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
	Function(const _fptr &f) : fptr(f) {}

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
};

// Allocate function
template <class T, class... Args>
Function new_(Args ... args)
{
	_function *f = new T(args...);
	return Function(_fptr(f));
}

// Get function
struct Get : public _function {
	int index;

	Get(int i) : _function(1, op_get), index(i) {}
};

// Constant getter
struct Const : public _function {
	int index;

	Const(int i) : _function(1, op_const), index(i) {}
};

// Instruction sequence for a function
class ISeq : public _function {
	// Instructions are a sequence of functions
	std::vector <_function *> _instrs;

	// Variables
	//	must be filled out before computation
	std::vector <Constant> _vars;

	// Constants
	std::vector <Constant> _consts;

	// Load inputs to _vars
	void _load(const Input &ins) {
		// TODO: should make sure that the ins == inputs
		_vars = ins;
	}

	// Deal with special instructions
	bool _ispec(_function *ftn, std::stack <Constant> &ops) {
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
			index = reinterpret_cast <Get *> (ftn)->index;
			ops.push(_vars[index]);
			return true;
		case op_const:
			// Get index and push the corresponding constant
			index = reinterpret_cast <Get *> (ftn)->index;
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
	void _exec(_function *ftn, std::stack <Constant> &ops) {
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
protected:
	// Protected constructor
	ISeq(std::vector <_function *> instrs,
		std::vector <Constant> consts, 
		int nins) : _function(nins),
			_instrs(instrs),
			_consts(consts) {}
public:
	// Evaluate the sequence
	Constant compute(const Input &ins) {
		// Load inputs
		_load(ins);

		// Stack of operands
		std::stack <Constant> ops;

		// Execute all the instructions
		for (_function *ftn : _instrs)
			_exec(ftn, ops);
		
		// Return top-most value on the stack
		return ops.top();
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

// Variables are just placeholders
struct Variable {};

// TODO: variable should be a type of Iseq (with no instructions)

// Overloaded operators to create functions
Function operator+(const Variable &, const Variable &);
Function operator-(const Variable &, const Variable &);

}

}

#endif
