#include <variable.hpp>

namespace zhetapi {

// Constructors
Variable::Variable(Token *tptr, const std::string &str) : __symbol(str)
{
	__tptr.reset(tptr);
}

Variable::Variable(const Variable &other)
{
	__tptr = other.__tptr;
	__symbol = other.__symbol;
}

// Copy
Variable &Variable::operator=(const Variable &other)
{
	if (this != &other) {
		__tptr = other.__tptr;
		__symbol = other.__symbol;
	}

	return *this;
}

// Reference
std::shared_ptr <Token> &Variable::get()
{
	return __tptr;
}

const std::shared_ptr <Token> &Variable::get() const
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
		return __symbol + "\t[" + __tptr->str() + "]";
	
	return __symbol + "\t[nullptr]";
}

Token *Variable::copy() const
{
	return new Variable(__tptr->copy(), __symbol);
}

bool Variable::operator==(Token *tptr) const
{
	Variable *var = dynamic_cast <Variable *> (tptr);

	if (!var)
		return true;
	
	return (__symbol == var->__symbol) && (__tptr == var->__tptr);
}

}