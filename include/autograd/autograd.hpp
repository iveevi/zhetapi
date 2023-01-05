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

// Foward declarations
class Function;
class Variable;

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

// Really should only be ISeq and _variable
class Function {
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
		cs.push_back(_function::Ptr(new _repl_const(c, i)));
		_ftr_process(cs, ++i, args...);
	}

	static void _ftr_process(_function::Compositions &cs, int &i) {}

	// Process variadic arguments for permute
	template <class ... Args>
	static void _prm_process(std::vector <_variable *> &, const Variable &, Args ...);

	static void _prm_process(std::vector <_variable *> &) {}
	
	// Invoking methods
	_function::Property fn(const std::string &name, const _function::Arguments &args) {
		const auto &[f, mt] = fptr->method_table();

		auto it = mt.find(name);
		if (it == mt.end())
			throw std::runtime_error("_function: method " + name + " not found");

		return it->second(f, args);
	}

	// Check argument validity with variant
	template <class T, class U> struct is_one_of;

	template <class T, class... Ts> 
	struct is_one_of <T, std::variant<Ts...>>
		: std::bool_constant <(std::is_same_v <T, Ts> || ...)> { };
protected:
	// Function pointer
	_function::Ptr fptr;
public:
	// Constructors
	Function() : fptr(nullptr) {}
	// Function(_function *f) : fptr(f) {}
	Function(const _function::Ptr &f) : fptr(f) {}

	// Get raw and pointer handle
	_function *raw() const {
		return fptr.get();
	}

	const _function::Ptr &get() const {
		return fptr;
	}

	// As a convertion operator
	operator const _function::Ptr &() const {
		return fptr;
	}

	operator _function::Ptr &() {
		return fptr;
	}

	// Composition
	template <class ... Args, typename = typename std::enable_if
		<fret <Args...> ::compose> ::type>
	Function operator()(Args ... args) {
		_function::Compositions cs;
		int i = 0;

		_ftr_process(cs, i, args...);
		return fptr->compose(cs);
	}

	// Computation
	template <class ... Args, typename = typename std::enable_if
		<!fret <Args...> ::compose> ::type>
	Constant operator()(Args ... args) {
		_function::Input inputs;
		_cmp_process(inputs, args...);
		return fptr->compute(inputs);
	}

	Constant operator()(const _function::Input &inputs) {
		return fptr->compute(inputs);
	}
	
	// Invoking pseudo-virtual methods
	template <class ... Args>
	_function::Property fn(const std::string &name, Args ... args) {
		static_assert(
			(is_one_of <Args, _function::Property> ::value && ...),
			"Arguments must be compatible with _function::Property"
		);

		return fn(name, _function::Arguments {args...});
	}

	// Differentiation
	Function differentiate(const int i) const {
		return fptr->diff(i);
	}

	// Machine learning
	_function::Gradient gradient(const _function::Input &ins,
			const _function::Input &igrads) const {
		return fptr->gradient(ins, igrads);
	}

	void update_parameters(GradientQueue &grads) {
		fptr->update_parameters(grads);
	}

	// Permutation
	template <class ... Args>
	void refactor(Args ... args) {
		// Permutation is only for op_iseq
		if (fptr->spop != _function::op_iseq) {
			// TODO: make a more decent error message
			throw std::runtime_error("Permutation is only for op_iseq");
		}

		std::vector <const _variable *> vars;
		_prm_process(vars, args...);

		// Cast and permute
		ISeq *iseq = reinterpret_cast <ISeq *> (fptr.get());
		iseq->refactor(vars);
	}

	template <class ... Args>
	Function refactored(Args ... args) const {
		// Permutation is only for op_iseq
		if (fptr->spop != _function::op_iseq) {
			// TODO: make a more decent error message
			throw std::runtime_error("Permutation is only for op_iseq");
		}

		std::vector <const _variable *> vars;
		_prm_process(vars, args...);

		// Cast and permute
		const ISeq *iseq = reinterpret_cast <ISeq *> (fptr.get());
		return Function(iseq->refactor(vars));
	}

	// Info about parameters
	int parameters() const {
		return fptr->parameters();
	}

	int tunable_parameters() const {
		return fptr->tunable_parameters();
	}

	// TODO: differentiate with respect to Variable,
	// add another function, _diff_vid(const int)...

	// Summary of functiooon
	std::string summary() const {
		return fptr->summary();
	}
};

// Allocate function
template <class T, class... Args>
Function new_(Args ... args)
{
	return Function(new_ftn_ <T> (args...));
}

// Wrapper around _variable for convenience
class Variable : public Function {
public:
	// Constructors
	Variable() : Function(new_ftn_ <_variable> ()) {}
};

// Permute function
template <class ... Args>
void Function::_prm_process(std::vector <_variable *> &vars, const Variable &v, Args ... args)
{
	// Get variable
	assert(v.get()->spop == _function::op_var);
	vars.push_back(reinterpret_cast <_variable *> (v.raw()));
	_prm_process(vars, args...);
}

// Overloaded operators
Function operator+(const Function &, const Function &);
Function operator-(const Function &, const Function &);
Function operator*(const Function &, const Function &);
Function operator/(const Function &, const Function &);

Function operator+(const Function &, const Constant &);
Function operator-(const Function &, const Constant &);
Function operator*(const Function &, const Constant &);
Function operator/(const Function &, const Constant &);

Function operator+(const Constant &, const Function &);
Function operator-(const Constant &, const Function &);
Function operator*(const Constant &, const Function &);
Function operator/(const Constant &, const Function &);

// Function class generating macro
#define FUNCTION_CLASS(name, inputs, str)					\
	Constant _k##name(const _function::Input &);				\
										\
	_function::Ptr _diffk_##name(const int);				\
										\
	class _##name : public ISeq { 						\
	public: 								\
		struct kernel : public _function { 				\
			kernel() : _function(inputs) {} 			\
										\
			Constant compute(const Input &ins) override {		\
				return _k##name(ins);				\
			}							\
										\
			Ptr diff(const int i) const override 	{		\
				return _diffk_##name(i);			\
			}							\
										\
			std::string summary() const override {			\
				return str;					\
			}							\
		}; 								\
										\
		_##name() : ISeq(new_ftn_ <kernel> (), inputs) {} 			\
	};									\
										\
	extern Function name;

// Specialized function classes
FUNCTION_CLASS(sqrt, 1, "SQRT")
FUNCTION_CLASS(length, 1, "LENGTH")
FUNCTION_CLASS(exp, 1, "EXP")
FUNCTION_CLASS(log, 1, "LOG")
FUNCTION_CLASS(sin, 1, "SIN")
FUNCTION_CLASS(cos, 1, "COS")
FUNCTION_CLASS(tan, 1, "TAN")
FUNCTION_CLASS(square, 1, "SQUARE")
FUNCTION_CLASS(pow, 2, "POW")
FUNCTION_CLASS(dot, 2, "DOT")

FUNCTION_CLASS(flatten, 1, "FLATTEN")
FUNCTION_CLASS(reshape, 2, "RESHAPE")

}

}

#endif
