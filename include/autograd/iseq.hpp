#ifndef ZHETAPI_ISEQ_H_
#define ZHETAPI_ISEQ_H_

// Standard headers
#include <stack>
#include <unordered_map>
#include <vector>

// Library jeaders
#include "function.hpp"

#include "../io/print.hpp"

namespace zhetapi {

namespace autograd {

// Instruction sequence for a function
class ISeq : public _function {
public:
	// Public aliases
	// TODO: use this alias
	using Instructions = std::vector <const _function *>;
private:
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

	// Tree structure
	struct _node {
		const _function *fptr;
		std::vector <_node *> children;

		// Constructors
		_node(const _function *);
		_node(const _function *, const std::vector <_node *> &);

		// Printing the tree
		std::string str(int = 0) const;
	};

	// Optimization structures
	// TODO: reorganize all of this
	struct _cache_info {
		int refs = 0;
		_node *value;

		// Constructor
		_cache_info();
		_cache_info(int, _node *);
	};

	using _cache_map = std::unordered_map <int, _cache_info>;

	// Construct the tree structure
	_node *_tree(_cache_map &) const;

	void _tree_walk(const _function *, std::stack <_node *> &, _cache_map &) const;

	// Rebuilding the tree
	void _rebuild(const _node *, Instructions &, _cache_map &) const;
protected:
	// Kernel function and # of inputs
	ISeq(const _function *, int);
	
	// Protected constructors
	ISeq(std::vector <const _function *>,
		std::vector <Constant>, int);

	// Overload composition
	// void _compose_iseq(Instructions &, const ISeq *, const _function *, int &) const;

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
	Constant compute(const Input &ins) const override;

	// Optimize the ISeq
	// TODO: protected
	void optimize();

	// Copy generator
	_function *copy() const override;

	// Dump instructions for debugging
	std::string summary() const override;
};

}

}

#endif
