#include <core/node_reference.hpp>

namespace zhetapi {

bool node_reference::address = true;

node_reference::node_reference(node *ref, const ::std::string &str, size_t
		idx, bool var) : _ref(ref), _symbol(str),
	_index(idx), _var(var) {}

void node_reference::set(node *nptr)
{
	_ref = nptr;
}

node *node_reference::get()
{
	return _ref;
}

const node &node_reference::get() const
{
	return *_ref;
}

size_t node_reference::index() const
{
	return _index;
}

const ::std::string &node_reference::symbol() const
{
	return _symbol;
}

bool node_reference::is_variable() const
{
	return _var;
}

Token::type node_reference::caller() const
{
	return Token::ndr;
}

Token *node_reference::copy() const
{
	return new node_reference(_ref, _symbol, _index, _var);
}

std::string node_reference::dbg_str() const
{
	std::ostringstream oss;

	if (address)
		oss << "ndr-\"" << _symbol << "\" points to " << _ref;
	else
		oss << "ndr-\"" << _symbol << "\"";

	return oss.str();
}

bool node_reference::operator==(Token *tptr) const
{
	node_reference *ndr = dynamic_cast <node_reference *> (tptr);

	if (ndr == nullptr)
		return false;

	return (ndr->_symbol == _symbol);
}

}
