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

struct _node;

// Cache structure for automatic differentiation
using Cache = std::unordered_map <_function *, _function::Input>;
using Node = std::shared_ptr <_node>;

// Tree structure
struct _node {
	_function::Ptr fptr;

	// TODO: should be a vector of plain nodes
	std::vector <Node> children;

	// Constructors
	_node(const _function::Ptr &);
	_node(const _function::Ptr &, const std::vector <Node> &);

	static Node make(const _function::Ptr &);
	static Node make(const _function::Ptr &, const std::vector <Node> &);
	
	// Printing the tree
	std::string str(int = 0) const;
};

// Instruction sequence for a function
class ISeq : public _function {
public:
	// Public aliases
	using Instructions = std::vector <_function::Ptr>;
	using ConstantCache = std::vector <Constant>;
private:
	// Private aliases
	using Var = std::shared_ptr <_variable>;
	using Variables = std::vector <Var>;

	// Information about cache usage, for optimization
	struct _cache_info {
		int refs = 0;
		Node value;

		// Constructor
		_cache_info();
		_cache_info(int, const Node &);
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
	
	// TODO: should really be finput = nullptr
	// TODO: actually cached input is currently obsolete
	Input			_cached_in;
	Cache			_cached_finputs;

	// TODO: cache tree?

	// Append helpers
	void append_variable(const _variable *);
	void append_iseq(const ISeq *const);
	int index_of(const _variable *) const;
	void _append_function(const Ptr &);

	template <class ... Args>
	void _append(const _function::Ptr &, Args ...);

	// Computation helpers
	void _load(const Input &);
	void storec(std::stack <Constant> &, int) const;
	bool _ispec(const Ptr &, std::stack <Constant> &);
	void _exec(const Ptr &, std::stack <Constant> &);

	// Composing functions and variables
	_function::Ptr _compose(const Compositions &) const override;

	// Tree building and rebuilding
	Node _tree(_cache_map &) const;
	void _tree_walk(const Ptr &, std::stack <Node> &,
		_cache_map &) const;
	void _rebuild(const Node &, Instructions &,
		ConstantCache &, _cache_map &,
		const ConstantCache &) const;

	// Optimization functions
	void _optimize();
	_reindex_map _generate_reindex_map() const;

	// TODO: remove const
	// Differentiation functions
	friend Node _diff_tree(const Node &, int);

	_function::Ptr diff(const int) const override;
protected:
	// Protected constructors
	ISeq(const _function::Ptr &, int);
	ISeq(const std::vector <_function::Ptr> &,
		std::vector <Constant>, int);
	ISeq(const std::vector <_function::Ptr> &,
		std::vector <Constant>, int,
		const _reindex_map &);

	std::pair <_function *, const MethodTable &> method_table() override;
public:
	// TODO: check function to make sure only
	// one element remains on the stack

	// Empty constructor
	ISeq();

	// Get a variable
	// TODO: protected?
	const Var &get(int) const;

	// Append a sequence of instructions
	template <class ... Args>
	void append(Args ...);

	// Evaluate the sequence
	Constant compute(const Input &) override;

	// Evaluate gradient
	Gradient gradient(const Input &, const Input &) override;

	// Apply gradients
	void update_parameters(GradientQueue &) override;

	// Permute the order of variables
	void refactor(const std::vector <const _variable *> &);
	_function::Ptr refactor(const std::vector <const _variable *> &) const;

	// Info about parameters
	int parameters() const override;
	int tunable_parameters() const override;

	// Dump instructions for debugging
	std::string summary() const override;
};

// Append a sequence of instructions
template <class ... Args>
void ISeq::_append(const _function::Ptr &fptr, Args ... args)
{
	_append_function(fptr);
	if constexpr (sizeof ... (args) > 0)
		_append(args ...);
}

template <class ... Args>
void ISeq::append(Args ... args)
{
	// Append all, then optimize
	_append(args...);
	_optimize();
}


}

}

#endif
