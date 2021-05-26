#include <core/node_list.hpp>
#include <core/node_manager.hpp>
#include <core/collection.hpp>

namespace zhetapi {

node_list::node_list(const std::vector <node> &nodes)
		: _nodes(nodes) {}

Token *node_list::evaluate(Engine *context) const
{
	std::vector <Token *> toks;
	for (node nd : _nodes) {
		node_manager nm(context, nd);

		toks.push_back(nm.value(context));
	}

	// TODO: clear memory
	return new Collection(toks);
}

Token::type node_list::caller() const
{
	return Token::token_node_list;
}

Token *node_list::copy() const
{
	return new node_list(_nodes);
}

std::string node_list::dbg_str() const
{
	return "node_list of " + std::to_string(_nodes.size()) + " nodes";
}

bool node_list::operator==(Token *tptr) const
{
	node_list *ndl = dynamic_cast <node_list *> (tptr);

	size_t n = ndl->_nodes.size();
	if (ndl == nullptr || n != ndl->_nodes.size())
		return false;

	for (size_t i = 0; i < n; i++) {
		if (&(_nodes[i]) != &(ndl->_nodes[i]))
			return false;
	}

	return true;
}

}
