#include <core/node_differential.hpp>

#include <core/node_differential.hpp>

namespace zhetapi {

node_differential::node_differential(Token *tptr) : __ref(tptr) {}

Token::type node_differential::caller() const
{
	return Token::ndd;
}

Token *node_differential::copy() const
{
	return new node_differential(__ref->copy());
}

std::string node_differential::str() const
{
	return "differential \"" + __ref->str() + "\"";
}

bool node_differential::operator==(Token *tptr) const
{
	node_differential *ndd = dynamic_cast <node_differential *> (tptr);

	if (ndd == nullptr)
		return false;

	return (*__ref == ndd->__ref);
}

}

