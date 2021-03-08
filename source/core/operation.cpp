#include <core/operation.hpp>

namespace zhetapi {

operation::operation() {}

operation::operation(
		const std::string &in,
		const std::string &out,
		std::size_t opers,
		mapper fopn)
		: __input(in), __output(out),
		__ops(opers), __opn(fopn) {}

Token *operation::compute(const std::vector <Token *> &ins) const
{
	if (ins.size() != __ops)
		throw bad_input_size();

	return __opn(ins);
}

std::string operation::str() const
{
	return "[" + __input + "](" + std::to_string(__ops)
		+ ") - [" + __output + "]";
}

Token::type operation::caller() const
{
	return Token::opn;
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

	return (__ops == optr->__ops)
		&& (__input == optr->__input)
		&& (__output == optr->__output);
}

}
