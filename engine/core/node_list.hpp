#ifndef NODE_LIST_H_
#define NODE_LIST_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <token.hpp>

#include <core/node.hpp>

namespace zhetapi {

class Engine;

class node_list : public Token {
protected:
	std::vector <node>	_nodes;
public:
	node_list(const std::vector <node> &);

	Token *evaluate(Engine *engine) const;

	type caller() const override;
	Token *copy() const override;
	std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;
};

}

#endif
