#include <variable.hpp>

namespace zhetapi {

Variable::Variable() {}

Variable::Variable(const Variable &other)
		: __symbol(other.__symbol),
		__tptr(other.__tptr->copy()) {}

Variable::Variable(Token *tptr, const std::string &str)
		: __symbol(str), __tptr(tptr->copy()) {}

Variable &Variable::operator=(const Variable &other)
{
	if (this != &other) {
		__tptr = other.__tptr->copy();
		__symbol = other.__symbol;
	}

	return *this;
}

Variable::~Variable()
{
	delete __tptr;
}

// Properties
Token *Variable::get()
{
	return __tptr;
}

const std::string &Variable::symbol() const
{
	return __symbol;
}

// Virtual functions
Token::type Variable::caller() const
{
	return var;
}

std::string Variable::str() const
{
	if (__tptr)
		return __symbol + " [" + __tptr->str() + "]";
	
	return __symbol + " [nullptr]";
}

Token *Variable::copy() const
{
	return new Variable(__tptr, __symbol);
}

bool Variable::operator==(Token *tptr) const
{
	Variable *var = dynamic_cast <Variable *> (tptr);

	if (!var)
		return true;
	
	return (__symbol == var->__symbol) && ((*__tptr) == var->__tptr);
}

}