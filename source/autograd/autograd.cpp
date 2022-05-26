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
	inline _function *_diffk_##name(const int vindex)

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
		new _function(2, _function::op_add)
	);

	return Function(iseq);
}

Function operator-(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new _function(2, _function::op_sub)
	);

	return Function(iseq);
}

Function operator*(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new _function(2, _function::op_mul)
	);

	return Function(iseq);
}

Function operator/(const Function &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(), rhs.get(),
		new _function(2, _function::op_div)
	);

	return Function(iseq);
}

// With constants
Function operator+(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new _repl_const(rhs, -1),
		new _function(2, _function::op_add)
	);

	return Function(iseq);
}

Function operator-(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new _repl_const(rhs, -1),
		new _function(2, _function::op_sub)
	);

	return Function(iseq);
}

Function operator*(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new _repl_const(rhs, -1),
		new _function(2, _function::op_mul)
	);

	return Function(iseq);
}

Function operator/(const Function &lhs, const Constant &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		lhs.get(),
		new _repl_const(rhs, -1),
		new _function(2, _function::op_div)
	);

	return Function(iseq);
}

// Reverse operators
Function operator+(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new _repl_const(lhs, -1),
		rhs.get(),
		new _function(2, _function::op_add)
	);

	return Function(iseq);
}

Function operator-(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new _repl_const(lhs, -1),
		rhs.get(),
		new _function(2, _function::op_sub)
	);

	return Function(iseq);
}

Function operator*(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new _repl_const(lhs, -1),
		rhs.get(),
		new _function(2, _function::op_mul)
	);

	return Function(iseq);
}

Function operator/(const Constant &lhs, const Function &rhs)
{
	ISeq *iseq = new ISeq();
	iseq->append(
		new _repl_const(lhs, -1),
		rhs.get(),
		new _function(2, _function::op_div)
	);

	return Function(iseq);
}

// Standard function kernels
KERNEL(sqrt)
{
	return ins[0].transform(sqrtl);
}

KERNEL(norm)
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
		new Get(0),
		new _function(2, _function::op_mul),
		new _repl_const(2.0, -1),
		new _function(2, _function::op_mul)
	);

	return iseq;
}

DERIVATIVE(norm)
{
	ISeq *iseq = new ISeq();

	// norm(f(x)) -> f'(x) / norm(f(x))
	iseq->append(
		_iop::differential(0),
		new Get(0),
		new _norm::kernel(),
		new _function(2, _function::op_div)
	);

	return iseq;
}

DERIVATIVE(sqrt)
{
	ISeq *iseq = new ISeq();

	// sqrt(f(x)) -> f'(x) / 2 * sqrt(f(x))
	iseq->append(
		_iop::differential(0),
		new _repl_const(2.0, -1),
		new Get(0),
		new _sqrt::kernel(),
		new _function(2, _function::op_mul),
		new _function(2, _function::op_div)
	);

	return iseq;
}

// TODO: fill out the rest
DERIVATIVE(exp)
{
	ISeq *iseq = new ISeq();

	// e^f(x) -> f'(x) * e^f(x)
	iseq->append(
		new Get(0),
		new _exp::kernel(),
		_iop::differential(0),
		new _function(2, _function::op_mul)
	);

	return iseq;
}

DERIVATIVE(log)
{
	ISeq *iseq = new ISeq();

	// log(f(x)) -> f'(x) / f(x)
	iseq->append(
		_iop::differential(0),
		new Get(0),
		new _log::kernel(),
		new _function(2, _function::op_div)
	);

	return iseq;
}

DERIVATIVE(sin)
{
	ISeq *iseq = new ISeq();

	// sin(f(x)) -> f'(x) * cos(f(x))
	iseq->append(
		_iop::differential(0),
		new Get(0),
		new _cos::kernel(),
		new _function(2, _function::op_mul)
	);

	return iseq;
}

DERIVATIVE(cos)
{
	ISeq *iseq = new ISeq();

	// cos(f(x)) -> -f'(x) * sin(f(x))
	iseq->append(
		_iop::differential(0),
		new Get(0),
		new _sin::kernel(),
		new _function(2, _function::op_mul),
		new _repl_const(-1, -1),
		new _function(2, _function::op_mul)
	);

	return iseq;
}

// TODO: seq operation
DERIVATIVE(tan)
{
	ISeq *iseq = new ISeq();

	// tan(f(x)) -> f'(x) / (cos(f(x))^2)
	iseq->append(
		_iop::differential(0),
		new Get(0),
		new _cos::kernel(),
		new Get(0),
		new _cos::kernel(),
		new _function(2, _function::op_mul),
		_iop::differential(0),
		new _function(2, _function::op_div)
	);

	return iseq;
}

// TODO: also the case where f(x)^g(x) -> check this!
DERIVATIVE(pow)
{
	ISeq *iseq = new ISeq();

	// TODO: gotta test all thi<F2><F2>s

	// f(x)^e -> e * f(x)^(e-1) * f'(x)
	iseq->append(
		_iop::differential(0),
		new _repl_const(-1.0, -1),
		new Get(1),
		new _function(2, _function::op_sub),
		new Get(0),
		new _pow::kernel(),
		new _function(2, _function::op_mul)
	);

	return iseq;
}

DERIVATIVE(reshape)
{
	ISeq *iseq = new ISeq();

	// Same function, with all ones
	iseq->append(
		new _repl_const(1.0, -1),
		new Get(0)
	);

	return iseq;
}

}

}
