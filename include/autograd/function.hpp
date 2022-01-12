#ifndef ZHETAPI_FUNCTION_H_
#define ZHETAPI_FUNCTION_H_

// Standard headers
#include <stack>
#include <vector>

// Library headers
#include "../tensor.hpp"

namespace zhetapi {

namespace autograd {

// Constants are just tensors
using Constant = Tensor <long double>;

// Basic structure of a function
struct Function {
	// Special operations
	enum Special {
		op_none = -1,
		op_get,
		op_const
	};

	// Special operation code (defualt -1 = none)
	int spop = op_none;

	int inputs = -1;

	// Type of input
	using Input = std::vector <Constant>;

	// Constructors
	Function(int nins, int op = op_none) : inputs(nins), spop(op) {}

	// Not pure virtual so that special operations
	//	can get away without implementing it
	virtual Constant compute(const Input &) {
		return Constant();
	}

	// Operator overloading and variadics
	Constant operator()(const Input &ins) {
		return this->compute(ins);
	}
};

// Get function
struct Get : public Function {
	int index;

	Get(int i) : Function(1, op_get), index(i) {}
};

// Constant getter
struct Const : public Function {
	int index;

	Const(int i) : Function(1, op_const), index(i) {}
};

// Instruction sequence for a function
class ISeq : public Function {
	// Instructions are a sequence of functions
	std::vector <Function *> _instrs;

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
	bool _ispec(Function *ftn, std::stack <Constant> &ops) {
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
		default:
			break;
		}

		return false;
	}

	// Execute an instruction
	void _exec(Function *ftn, std::stack <Constant> &ops) {
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
public:
	// Evaluate the sequence
	Constant compute(const Input &ins) {
		// Stack of operands
		std::stack <Constant> ops;

		// Execute all the instructions
		for (Function *ftn : _instrs)
			_exec(ftn, ops);
		
		// Return top-most value on the stack
		return ops.top();
	}
};

// Example: x^2 + x
//	becomes get(0), square, get(0), plus

}

}

#endif