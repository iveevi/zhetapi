#include "../include/autograd/autograd.hpp"

// Macros
#define KERNEL(name)						\
	Function name = new_ <_##name> ();			\
	inline Constant _k##name(const _function::Input &ins)

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

// Standard function kernels
KERNEL(sqrt)
{
	return ins[0].transform(sqrtl);
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

KERNEL(pow)
{
	// Use only the first element
	long double e = ins[1].get(0);

	return ins[0].transform(
		[e](long double x) -> long double {
			return powl(x, e);
		}
	);
}

}

}
