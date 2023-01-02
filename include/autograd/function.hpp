#ifndef ZHETAPI_FUNCTION_H_
#define ZHETAPI_FUNCTION_H_

// Standard headers
#include <deque>
#include <variant>
#include <vector>

// Library headers
#include "gradient_queue.hpp"

namespace zhetapi {

namespace autograd {

// Basic structure of a function
class _function {
public:
	// Special operations
	// TODO: keep output of this scope
	enum Special {
		op_none = -1,

		// Key ISeq operations
		op_get,
		op_const,
		op_repl_const,
		op_store_cache,
		op_get_cache,
		op_differential,

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
protected:
	// String versions of operations
	static constexpr const char *_spec_strs[] {
		"GET", "CONST", "REPLACE CONST",
		"STORE CACHE", "GET CACHE",
		"DIFFERENTIAL",

		"VAR", "ISEQ",

		"ADD", "SUB",
		"MUL", "DIV"
	};

	// By default, function composition returns null
	virtual _function *_compose(const Compositions &) const {
		return nullptr;
	}
public:
	// Constructors
	_function(int nins, int op = op_none) : inputs(nins), spop(op) {}

	// Virtual destructor for clean up of derived classes
	virtual ~_function() {}

	// Not pure virtual so that special operations
	//	can get away without implementing it
	virtual Constant compute(const Input &) {
		// TODO: separate into _compute, like compose?
		// then we can check size
		return Constant();
	}

	// Wrapper around _compose that checks #inputs
	_function *compose(const Compositions &cs) const {
		// TODO: add argsize exception
		if (cs.size() != inputs) {
			std::cout << "summary: " << summary() << std::endl;
			throw std::runtime_error("_function::compose size mismatch");
		}

		return this->_compose(cs);
	}

	// By default, function differentiation returns null
	virtual _function *diff(const int) const {
		return nullptr;
	}

	// Gradient packet
	struct Gradient {
		Input		igrads;	// inputs
		GradientQueue	grads;	// weights
	};

	// Machine learning - by default, no gradient
	// TODO: do we need a vector of igrads?
	virtual Gradient gradient(const Input &, const Input &) {
		// NOTE: no inputs are provided (only igrad)
		// inputs and any other information from the forward
		// pass must be cached by the functions
		return {};
	}

	// Apply gradients
	virtual void update_parameters(GradientQueue &) {}

	// Info about parameters
	virtual int parameters() const {
		// This is the number of tensor parameters
		return 0;
	}

	virtual int tunable_parameters() const {
		// This is the number of individual (scalar) parameters
		return 0;
	}

	// Copy pointer
	virtual _function *copy() const {
		return new _function(inputs, spop);
	}

	// Pseudo-virtual methods for function properties
	using Property = std::variant <bool, int, float>;
	using Arguments = std::vector <Property>;
	
	using Method = std::function <Property (_function *, const Arguments &)>;
	using MethodTable = std::unordered_map <std::string, Method>;

	// Method table for function properties
	virtual std::pair <_function *, const MethodTable &> method_table() {
		const MethodTable _map {};
		return {this, _map};
	}

	// Summary for each function
	virtual std::string summary() const {
		if (spop > op_none)
			return _spec_strs[spop];

		return "_function(#inputs = "
			+ std::to_string(inputs) + ")";
	}
};

using Gradient = _function::Gradient;

// Get function
// TODO: underscore convention?
struct Get : public _function {
	int index;

	Get(int i) : _function(1, op_get), index(i) {}

	// Gradient
	Gradient gradient(const Input &, const Input &x) override {
		Gradient g;
		g.igrads.push_back(x[index]);
		return g;
	}

	// Copy pointer
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

	Const(int i) : _function(0, op_const), index(i) {}

	// Overload summary to include index
	std::string summary() const override {
		return "CONST (" + std::to_string(index) + ")";
	}
};

// Replace a variable with a constant
struct _repl_const : public _function {
	int index; // TODO: what is this for?

	Constant value;

	_repl_const(const Constant &v, int i)
		: _function(0, op_repl_const), value(v), index(i) {}

	// Overload summary to show index
	std::string summary() const override {
		return "REPLACE CONST (" + std::to_string(index) + ")";
	}
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

// Operation with index
struct _iop : public _function {
	int index;

	_iop(int i, int nins, int spop) : _function(nins, spop), index(i) {}

	// Overload summary to include index
	_function *copy() const override {
		return new _iop(index, inputs, spop);
	}

	// Overload summary to include index
	std::string summary() const override {
		return std::string(_spec_strs[spop])
			+ " (" + std::to_string(index) + ")";
	}

	// Factories
	static _function *differential(int i) {
		return new _iop(i, 0, op_differential);
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
	_variable(int x) : _function(0, op_var), id(x) {}

	_function *copy() const override {
		return new _variable(id, value);
	}

	// Overload summary to include id
	std::string summary() const override {
		return "variable (id: " + std::to_string(id) + ")";
	}
};

}

}

#endif
