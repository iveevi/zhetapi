#include "../inc/node_reference.hpp"

namespace zhetapi {

	bool node_reference::address = true;

	node_reference::node_reference(node *ref, const std::string &str, size_t
			idx, bool var) : __ref(ref), __symbol(str),
		__index(idx), __var(var) {}

	node *node_reference::get()
	{
		return __ref;
	}

	const node &node_reference::get() const
	{
		return *__ref;
	}

	size_t node_reference::index() const
	{
		return __index;
	}

	const std::string &node_reference::symbol() const
	{
		return __symbol;
	}

	bool node_reference::is_variable() const
	{
		return __var;
	}

	Token::type node_reference::caller() const
	{
		return Token::ndr;
	}

	Token *node_reference::copy() const
	{
		return new node_reference(__ref, __symbol, __index, __var);
	}

	std::string node_reference::str() const
	{
		std::ostringstream oss;

		if (address)
			oss << "\"" << __symbol << "\" points to " << __ref;
		else
			oss << "\"" << __symbol << "\"";

		return oss.str();
	}

	bool node_reference::operator==(Token *tptr) const
	{
		node_reference *ndr = dynamic_cast <node_reference *> (tptr);

		if (ndr == nullptr)
			return false;

		return ndr->__symbol == __symbol;
	}

}