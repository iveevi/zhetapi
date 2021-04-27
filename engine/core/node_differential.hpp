#ifndef NODE_DIFFERENTIAL_H_
#define NODE_DIFFERENTIAL_H_

// C/C++ headers
#include <sstream>

// Engine headers
#include <core/node.hpp>

namespace zhetapi {

class node_differential : public Token {
protected:
	Token *		_ref;
public:
	explicit node_differential(Token *);

	Token *get() const;

	type caller() const override;
	Token *copy() const override;
	std::string dbg_str() const override;
	virtual bool operator==(Token *) const override;
};

}

#endif
