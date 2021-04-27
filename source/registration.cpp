#include <registration.hpp>

namespace zhetapi {

Registrable::Registrable() : _ident(""), _ftn(0) {}

Registrable::Registrable(const Registrable &other)
{
	_ftn = other._ftn;
	_ident = other._ident;
}

Registrable::Registrable(const ::std::string &ident, mapper ftn) : _ident(ident), _ftn(ftn) {}

Token *Registrable::operator()(const ::std::vector <Token *> &ins) const
{
	return _ftn(ins);
}

std::string Registrable::dbg_str() const
{
	return _ident;
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

	return _ident == reg->_ident;
}

}
