#include <core/operation.hpp>

namespace zhetapi {

	operation::operation() : __input(""), __output(""), ops(0) {}

	operation::operation(const operation &other)
	{
		__input = other.__input;
		__output = other.__output;
		ops = other.ops;
		__opn = other.__opn;
	}

	operation::operation(const ::std::string &in, const ::std::string &out, ::std::size_t
			opers, mapper fopn) : __input(in), __output(out), ops(opers),
			__opn(fopn) {}

	Token *operation::operator()(const ::std::vector <Token *> &ins) const
	{
		if (ins.size() != ops)
			throw count_mismatch();
		return __opn(ins);
	}

	::std::string operation::fmt() const
	{
		return __input;
	}

	::std::string operation::str() const
	{
		return "[" + __input + "](" + ::std::to_string(ops)
			+ ") - [" + __output + "]";
	}

	::std::size_t operation::inputs() const
	{
		return ops;
	}

	Token::type operation::caller() const
	{
		return opn;
	}

	Token *operation::copy() const
	{
		return new operation(*this);
	}

	bool operation::operator==(Token *t) const
	{
		operation *optr = dynamic_cast <operation *> (t);
		if (optr == nullptr)
			return false;

		return (ops == optr->ops)
			&& (__input == optr->__input)
			&& (__output == optr->__output);
	}

}
