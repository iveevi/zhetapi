#include <registration.hpp>

namespace zhetapi {

	Registrable::Registrable() : __ident(""), __ftn(0) {}

	Registrable::Registrable(const Registrable &other)
	{
		__ftn = other.__ftn;
		__ident = other.__ident;
	}

	Registrable::Registrable(const ::std::string &ident, mapper ftn) : __ident(ident), __ftn(ftn) {}

	Token *Registrable::operator()(const ::std::vector <Token *> &ins) const
	{
		return __ftn(ins);
	}

	::std::string Registrable::str() const
	{
		return __ident;
	}

	Token::type Registrable::caller() const
	{
		return reg;
	}

	Token *Registrable::copy() const
	{
		return new Registrable(*this);
	}

	bool Registrable::operator==(Token *t) const
	{
		Registrable *reg = dynamic_cast <Registrable *> (t);
		if (reg == nullptr)
			return false;

		return __ident == reg->__ident;
	}

}