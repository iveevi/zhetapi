#ifndef ZHETAPI_ISEQ_H_
#define ZHETAPI_ISEQ_H_

// Standard headers
#include <vector>

// Library jeaders
#include "function.hpp"

#include "../io/print.hpp"

namespace zhetapi {

namespace autograd {

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
	void _load(const Input &) const;

	// Executing instructions
	static Constant getc(std::stack <Constant> &);
	void storec(std::stack <Constant> &, int) const;
	bool _ispec(const _function *, std::stack <Constant> &) const;
	void _exec(const _function *, std::stack <Constant> &) const;

	// Get index of variable
	int index_of(_variable *);

	// Append helpers
	void append_variable(_variable *);
	void append_iseq(ISeq *);
protected:
	// Kernel function and # of inputs
	ISeq(const _function *, int);
	
	// Protected constructors
	ISeq(std::vector <const _function *>,
		std::vector <Constant>, int);

	// Overload composition
	_function *_compose(const Compositions &) const override;
public:
	// TODO: check function to make sure only
	// one element remains on the stack

	// Empty constructor
	ISeq();

	// Inserting instructions and functions
	void append(const _function *);

	// Append a sequence of instructions
	template <class ... Args>
	void append(const _function *fptr, Args ...);

	// Evaluate the sequence
	Constant compute(const Input &ins) const;

	_function *copy() const;

	// Dump instructions for debugging
	std::string summary() const;
};

}

}

#endif