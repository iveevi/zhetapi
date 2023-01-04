// Standard headers
#include <deque>
#include <map>

// Library headers
#include "../../include/autograd/autograd.hpp"

// Macros
#define KERNEL(name)						\
	Function name = new_ <_##name> ();			\
	inline Constant _k##name(const _function::Input &ins)

#define DERIVATIVE(name)					\
	inline _function::Ptr _diffk_##name(const int vindex)

namespace zhetapi {

namespace autograd {

// Static variables
constexpr const char *_function::_spec_strs[];

// Operators
Function operator+(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new_ftn_ <_function> (2, _function::op_add)
	);

	return Function(_function::Ptr(iseq));
}

Function operator-(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new_ftn_ <_function> (2, _function::op_sub)
	);

	return Function(_function::Ptr(iseq));
}

Function operator*(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return Function(_function::Ptr(iseq));
}

Function operator/(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new_ftn_ <_function> (2, _function::op_div)
	);

	return Function(_function::Ptr(iseq));
}

// With constants
Function operator+(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new_ftn_ <_repl_const> (rhs, -1),
		new_ftn_ <_function> (2, _function::op_add)
	);

	return Function(_function::Ptr(iseq));
}

Function operator-(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new_ftn_ <_repl_const> (rhs, -1),
		new_ftn_ <_function> (2, _function::op_sub)
	);

	return Function(_function::Ptr(iseq));
}

Function operator*(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new_ftn_ <_repl_const> (rhs, -1),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return Function(_function::Ptr(iseq));
}

Function operator/(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new_ftn_ <_repl_const> (rhs, -1),
		new_ftn_ <_function> (2, _function::op_div)
	);

	return Function(_function::Ptr(iseq));
}

// Reverse operators
Function operator+(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new_ftn_ <_repl_const> (lhs, -1),
		rhs.get(),
		new_ftn_ <_function> (2, _function::op_add)
	);

	return Function(_function::Ptr(iseq));
}

Function operator-(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new_ftn_ <_repl_const> (lhs, -1),
		rhs.get(),
		new_ftn_ <_function> (2, _function::op_sub)
	);

	return Function(_function::Ptr(iseq));
}

Function operator*(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new_ftn_ <_repl_const> (lhs, -1),
		rhs.get(),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return Function(_function::Ptr(iseq));
}

Function operator/(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new_ftn_ <_repl_const> (lhs, -1),
		rhs.get(),
		new_ftn_ <_function> (2, _function::op_div)
	);

	return Function(_function::Ptr(iseq));
}

// Standard function kernels
KERNEL(sqrt)
{
	return ins[0].transform(sqrtl);
}

KERNEL(length)
{
	return Constant {ins[0].length()};
}

KERNEL(exp)
{
	return ins[0].transform(expl);
}

KERNEL(log)
{
	return ins[0].transform(logl);
}

KERNEL(sin)
{
	return ins[0].transform(sinf);
}

KERNEL(cos)
{
	return ins[0].transform(cosf);
}

KERNEL(tan)
{
	return ins[0].transform(tanf);
}

KERNEL(square)
{
	return ins[0].transform(
		[](long double x) {
			return x * x;
		}
	);
}

KERNEL(pow)
{
	// Use only the first element
	long double e = ins[0].get(0);

	return ins[1].transform(
		[e](long double x) -> long double {
			return powl(x, e);
		}
	);
}

KERNEL(dot)
{
	return zhetapi::dot(ins[0], ins[1]);
}

KERNEL(flatten)
{
	auto copy = ins[0].copy();
	copy.flatten();
	return copy;
}

KERNEL(reshape)
{
	// Get shape from second element
	auto shape = ins[1].as_vector <size_t> ();
	auto copy = ins[0].copy();
	copy.reshape(shape);
	return copy;
}

// Standard function derivatives
DERIVATIVE(square)
{
	ISeq *iseq = new ISeq();

	// f(x)^2 -> 2 * f(x) * f'(x)
	iseq->append(
		_iop::differential(0),
		new_ftn_ <Get> (0),
		new_ftn_ <_function> (2, _function::op_mul),
		new_ftn_ <_repl_const> (2.0, -1),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return _function::Ptr(iseq);
}

DERIVATIVE(length)
{
	ISeq *iseq = new ISeq();

	// length(f(x)) -> f(x) * f'(x)
	iseq->append(
		_iop::differential(0),
		new_ftn_ <Get> (0),
		new_ftn_ <_function> (2, _function::op_mul),
		new_ftn_ <_repl_const> (1.0, -1),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return _function::Ptr(iseq);
}

DERIVATIVE(sqrt)
{
	ISeq *iseq = new ISeq();

	// sqrt(f(x)) -> f'(x) / 2 * sqrt(f(x))
	iseq->append(
		_iop::differential(0),
		new_ftn_ <_repl_const> (2.0, -1),
		new_ftn_ <Get> (0),
		new_ftn_ <_sqrt::kernel> (),
		new_ftn_ <_function> (2, _function::op_mul),
		new_ftn_ <_function> (2, _function::op_div)
	);

	return _function::Ptr(iseq);
}

// TODO: fill out the rest
DERIVATIVE(exp)
{
	ISeq *iseq = new ISeq();

	// e^f(x) -> f'(x) * e^f(x)
	iseq->append(
		new_ftn_ <Get> (0),
		new_ftn_ <_exp::kernel> (),
		_iop::differential(0),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return _function::Ptr(iseq);
}

DERIVATIVE(log)
{
	ISeq *iseq = new ISeq();

	// log(f(x)) -> f'(x) / f(x)
	iseq->append(
		_iop::differential(0),
		new_ftn_ <Get> (0),
		new_ftn_ <_log::kernel> (),
		new_ftn_ <_function> (2, _function::op_div)
	);

	return _function::Ptr(iseq);
}

DERIVATIVE(sin)
{
	ISeq *iseq = new ISeq();

	// sin(f(x)) -> f'(x) * cos(f(x))
	iseq->append(
		_iop::differential(0),
		new_ftn_ <Get> (0),
		new_ftn_ <_cos::kernel> (),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return _function::Ptr(iseq);
}

DERIVATIVE(cos)
{
	ISeq *iseq = new ISeq();

	// cos(f(x)) -> -f'(x) * sin(f(x))
	iseq->append(
		_iop::differential(0),
		new_ftn_ <Get> (0),
		new_ftn_ <_sin::kernel> (),
		new_ftn_ <_function> (2, _function::op_mul),
		new_ftn_ <_repl_const> (-1, -1),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return _function::Ptr(iseq);
}

// TODO: seq operation
DERIVATIVE(tan)
{
	ISeq *iseq = new ISeq();

	// tan(f(x)) -> f'(x) / (cos(f(x))^2)
	iseq->append(
		_iop::differential(0),
		new_ftn_ <Get> (0),
		new_ftn_ <_cos::kernel> (),
		new_ftn_ <Get> (0),
		new_ftn_ <_cos::kernel> (),
		new_ftn_ <_function> (2, _function::op_mul),
		_iop::differential(0),
		new_ftn_ <_function> (2, _function::op_div)
	);

	return _function::Ptr(iseq);
}

// TODO: also the case where f(x)^g(x) -> check this!
DERIVATIVE(pow)
{
	ISeq *iseq = new ISeq();

	// TODO: gotta test all this

	// f(x)^e -> e * f(x)^(e-1) * f'(x)
	iseq->append(
		_iop::differential(0),
		new_ftn_ <_repl_const> (-1.0, -1),
		new_ftn_ <Get> (1),
		new_ftn_ <_function> (2, _function::op_sub),
		new_ftn_ <Get> (0),
		new_ftn_ <_pow::kernel> (),
		new_ftn_ <_function> (2, _function::op_mul)
	);

	return _function::Ptr(iseq);
}

DERIVATIVE(dot)
{
	// TODO: implementing this
	return nullptr;
}

DERIVATIVE(flatten)
{
	ISeq *iseq = new ISeq();

	// flatten(f(x)) -> f(x)
	iseq->append(
		new_ftn_ <_repl_const> (1.0, 1),
		new_ftn_ <Get> (0)
	);

	return _function::Ptr(iseq);
}

DERIVATIVE(reshape)
{
	ISeq *iseq = new ISeq();

	// reshape(f(x), shape) -> f(x)
	iseq->append(
		new_ftn_ <_repl_const> (1.0, 1),
		new_ftn_ <Get> (0)
	);

	return _function::Ptr(iseq);
}

}

}
