#include <core/node_list.hpp>
#include <core/node_manager.hpp>

namespace zhetapi {

node_list::node_list(const std::vector <node> &nodes)
		: __nodes(nodes) {}

Token *node_list::evaluate(Engine *engine) const
{
	std::vector <Token *> toks;
	for (node nd : __nodes) {
		node_manager nm(nd, engine);

		toks.push_back(nm.value());
	}

	return new Operand <std::vector <Token *>> (toks);
}

Token::type node_list::caller() const
{
	return Token::token_node_list;
}

Token *node_list::copy() const
{
	return new node_list(__nodes);
}

std::string node_list::str() const
{
	return "node_list of " + std::to_string(__nodes.size()) + " nodes";
}

bool node_list::operator==(Token *tptr) const
{
	node_list *ndl = dynamic_cast <node_list *> (tptr);

	size_t n = ndl->__nodes.size();
	if (ndl == nullptr || n != ndl->__nodes.size())
		return false;

	for (size_t i = 0; i < n; i++) {
		if (&(__nodes[i]) != &(ndl->__nodes[i]))
			return false;
	}

	return true;
}

}
