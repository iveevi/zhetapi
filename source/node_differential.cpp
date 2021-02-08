#include <core/node_differential.hpp>

#include <core/node_differential.hpp>

namespace zhetapi {

	bool node_differential::address = true;

	node_differential::node_differential(node *ref, const std::string &str, size_t
			idx, bool var) : __ref(ref), __symbol(str),
		__index(idx), __var(var) {}

	node *node_differential::get()
	{
		return __ref;
	}

	const node &node_differential::get() const
	{
		return *__ref;
	}

	size_t node_differential::index() const
	{
		return __index;
	}

	const ::std::string &node_differential::symbol() const
	{
		return __symbol;
	}

	bool node_differential::is_variable() const
	{
		return __var;
	}

	Token::type node_differential::caller() const
	{
		return Token::ndd;
	}

	Token *node_differential::copy() const
	{
		return new node_differential(__ref, __symbol, __index, __var);
	}

	std::string node_differential::str() const
	{
		return "differential " + __symbol;
	}

	bool node_differential::operator==(Token *tptr) const
	{
		node_differential *ndd = dynamic_cast <node_differential *> (tptr);

		if (ndd == nullptr)
			return false;

		return (ndd->__symbol == __symbol);
	}
}

