#include <core/operation.hpp>

namespace zhetapi {

operation::operation() {}

operation::operation(
		const std::string &in,
		const std::string &out,
		std::size_t opers,
		mapper fopn)
		: _input(in), _output(out),
		_ops(opers), _opn(fopn) {}

Token *operation::compute(const std::vector <Token *> &ins) const
{
	if (ins.size() != _ops)
		throw bad_input_size();

	return _opn(ins);
}

std::string operation::str() const
{
	return "[" + _input + "](" + std::to_string(_ops)
		+ ") - [" + _output + "]";
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

	return (_ops == optr->_ops)
		&& (_input == optr->_input)
		&& (_output == optr->_output);
}

}
