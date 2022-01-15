#ifndef ZHETAPI_AUTOGRAD_H_
#define ZHETAPI_AUTOGRAD_H_

// Standard headers
#include <cmath>
#include <memory>
#include <sstream>
#include <stack>

// Library headers
#include "function.hpp"
#include "iseq.hpp"

namespace zhetapi {

namespace autograd {

// Function alias is a wrapper around shared ptr
using _fptr = std::shared_ptr <_function>;

// Foward declarations
// TODO: at the top
class Function;

// Templated return type
//	to distinguish between
//	Cosntant or Function return
template <class T, class ... Args>
struct fret_helper {
	static constexpr bool compose = std::is_base_of <Function, T> ::value
		|| fret_helper <Args...> ::compose;
};

template <class T>
struct fret_helper <T> {
	static constexpr bool compose = std::is_base_of <Function, T> ::value;
};

template <class ... Args>
struct fret {
	static constexpr bool compose = fret_helper <Args...> ::compose;
};

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

	static void _cmp_process(_function::Input &input) {}

	// Process variadic arguments for compose
	template <class ... Args>
	static void _ftr_process(_function::Compositions &cs, int &i, const Function &f, Args ... args) {
		cs.push_back(f.get());
		_ftr_process(cs, ++i, args...);
	}
	
	template <class ... Args>
	static void _ftr_process(_function::Compositions &cs, int &i, const Constant &c, Args ... args) {
		cs.push_back(new _repl_const(c, i));
		_ftr_process(cs, ++i, args...);
	}

	static void _ftr_process(_function::Compositions &cs, int &i) {}
public:
	// Constructors
	Function() : fptr(nullptr) {}
	Function(_function *f) : fptr(f) {}
	Function(const _fptr &f) : fptr(f) {}

	// Get raw handle
	_function *get() const {
		return fptr.get();
	}

	template <class ... Args, typename = typename std::enable_if <fret <Args...> ::compose> ::type>
	Function operator()(Args ... args) {
		_function::Compositions cs;
		int i = 0;

		_ftr_process(cs, i, args...);
		return fptr->compose(cs);
	}
	
	template <class ... Args, typename = typename std::enable_if <!fret <Args...> ::compose> ::type>
	Constant operator()(Args ... args) {
		_function::Input inputs;
		_cmp_process(inputs, args...);
		return fptr->compute(inputs);
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

// Wrapper around _variable for convenience
class Variable : public Function {
public:
	// Constructors
	Variable() : Function(new_ <_variable> ()) {}
};

// Overloaded operators
Function operator+(const Function &, const Function &);
Function operator-(const Function &, const Function &);
Function operator*(const Function &, const Function &);
Function operator/(const Function &, const Function &);

// Function class generating macro
#define FUNCTION_CLASS(name, inputs, str)					\
	Constant _k##name(const _function::Input &);				\
										\
	class _##name : public ISeq { 						\
		struct kernel : public _function { 				\
			kernel() : _function(inputs) {} 			\
										\
			Constant compute(const Input &ins) const override {	\
				return _k##name(ins);				\
			}							\
										\
			std::string summary() const override {			\
				return str;					\
			}							\
										\
			_function *copy() const override {			\
				return new kernel();				\
			}							\
		}; 								\
	public: 								\
		_##name() : ISeq(new kernel(), inputs) {} 			\
	};									\
										\
	extern Function name;

// Specialized function classes
FUNCTION_CLASS(sqrt, 1, "SQRT")
FUNCTION_CLASS(exp, 1, "EXP")
FUNCTION_CLASS(log, 1, "LOG")
FUNCTION_CLASS(sin, 1, "SIN")
FUNCTION_CLASS(cos, 1, "COS")
FUNCTION_CLASS(tan, 1, "TAN")
FUNCTION_CLASS(pow, 2, "POW")

}

}

#endif
