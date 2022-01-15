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
	using Instructions = std::vector <const _function *>;
	using ConstantCache = std::vector <Constant>;
private:
	// Private aliases
	using Variables = std::vector <_variable *>;
	
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

	// Information about cache usage, for optimization
	struct _cache_info {
		int refs = 0;
		_node *value;

		// Constructor
		_cache_info();
		_cache_info(int, _node *);
	};

	using _cache_map = std::unordered_map <int, _cache_info>;

	// Reindexing map
	using _reindex_map = std::unordered_map <int, int>;

	// TODO: JIT function to compile into object code
	//	this should be possible since the types
	//	are homogenous

	// Member variables
	Instructions		_instrs;	// Instruction sequence
	Variables		_vars;		// Variables
	ConstantCache		_consts;	// Fixed constants
	mutable ConstantCache	_cache;		// More cache for flow of execution

	// TODO: cache tree?

	// Append helpers
	void append_variable(_variable *);
	void append_iseq(ISeq *);
	int index_of(_variable *);	// TODO: is this necessary?

	// Computation helpers
	void _load(const Input &) const;
	void storec(std::stack <Constant> &, int) const;
	bool _ispec(const _function *, std::stack <Constant> &) const;
	void _exec(const _function *, std::stack <Constant> &) const;

	// Composing functions and variables
	_function *_compose(const Compositions &) const override;

	// Optimization functions
	_node *_tree(_cache_map &) const;
	void _tree_walk(const _function *, std::stack <_node *> &,
		_cache_map &) const;
	void _rebuild(const _node *, Instructions &,
		_cache_map &) const;
	void _optimize();
protected:
	// Protected constructors
	ISeq(const _function *, int);
	ISeq(std::vector <const _function *>,
		std::vector <Constant>, int);
	ISeq(std::vector <const _function *>,
		std::vector <Constant>, int,
		const _reindex_map &);
public:
	// TODO: check function to make sure only
	// one element remains on the stack

	// Empty constructor
	ISeq();

	// Get a variable
	// TODO: protected?
	_variable *get(int) const;

	// Inserting instructions and functions
	void append(const _function *);

	// Append a sequence of instructions
	template <class ... Args>
	void append(const _function *fptr, Args ...);

	// Evaluate the sequence
	Constant compute(const Input &ins) const override;

	// Copy generator
	_function *copy() const override;

	// Dump instructions for debugging
	std::string summary() const override;
};

// Append a sequence of instructions
template <class ... Args>
void ISeq::append(const _function *fptr, Args ... args)
{
	append(fptr);
	append(args...);
}


}

}

#endif
