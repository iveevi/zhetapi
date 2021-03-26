#include <variable.hpp>

namespace zhetapi {

Variable::Variable() {}

Variable::Variable(const Variable &other)
		: _symbol(other._symbol),
		_tptr(other._tptr->copy()) {}

Variable::Variable(Token *tptr, const std::string &str)
		: _symbol(str), _tptr(tptr->copy()) {}

Variable &Variable::operator=(const Variable &other)
{
	if (this != &other) {
		_tptr = other._tptr->copy();
		_symbol = other._symbol;
	}

	return *this;
}

Variable::~Variable()
{
	delete _tptr;
}

// Properties
Token *Variable::get()
{
	return _tptr;
}

const std::string &Variable::symbol() const
{
	return _symbol;
}

// Virtual functions
Token::type Variable::caller() const
{
	return var;
}

std::string Variable::str() const
{
	if (_tptr)
		return _symbol + " [" + _tptr->str() + "]";
	
	return _symbol + " [nullptr]";
}

Token *Variable::copy() const
{
	return new Variable(_tptr, _symbol);
}

bool Variable::operator==(Token *tptr) const
{
	Variable *var = dynamic_cast <Variable *> (tptr);

	if (!var)
		return true;
	
	return (_symbol == var->_symbol) && ((*_tptr) == var->_tptr);
}

}