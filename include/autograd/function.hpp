#ifndef ZHETAPI_FUNCTION_H_
#define ZHETAPI_FUNCTION_H_

// Standard headers
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
		op_repl_const,
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
		"GET", "CONST", "REPLACE CONST",
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

	// Overload summary to include index
	std::string summary() const override {
		return "CONST (" + std::to_string(index) + ")";
	}
};

// Replace a variable with a constant
struct _repl_const : public _function {
	int index;

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

	// Overload summary to include id
	std::string summary() const override {
		return "variable (id: " + std::to_string(id) + ")";
	}
};

}

}

#endif